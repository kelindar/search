// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

// Package openai provides text embeddings through OpenAI-compatible HTTP APIs.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	defaultTimeout = 60 * time.Second
)

// Option configures a Client.
type Option func(*Client)

// BaseURL sets the API base URL. The embeddings endpoint is appended to it.
func BaseURL(value string) Option {
	return func(c *Client) { c.endpoint = value }
}

// Dimensions requests embeddings with the given dimension. A zero value omits
// the request field; negative values are rejected by New.
func Dimensions(value int) Option {
	return func(c *Client) { c.dimensions = value }
}

// Headers sets request headers, copying them when New applies the option.
// Authorization and Content-Type are always set by the client.
func Headers(value http.Header) Option {
	return func(c *Client) { c.headers = cloneHeaders(value) }
}

// HTTPClient sets the HTTP client used for requests.
func HTTPClient(value *http.Client) Option {
	return func(c *Client) { c.httpClient = value }
}

// Client sends embedding requests to OpenAI. A Client is safe for concurrent
// use after construction.
type Client struct {
	apiKey     string
	model      string
	endpoint   string
	dimensions int
	headers    http.Header
	httpClient *http.Client
}

// New constructs a client with the OpenAI /v1 base URL, provider-default
// dimensions, no extra headers, and a 60-second HTTP timeout.
func New(apiKey, model string, options ...Option) (*Client, error) {
	cfg := &Client{
		apiKey:     apiKey,
		model:      model,
		endpoint:   defaultBaseURL,
		headers:    make(http.Header),
		httpClient: &http.Client{Timeout: defaultTimeout},
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("openai: nil option")
		}
		option(cfg)
	}

	switch {
	case strings.TrimSpace(cfg.apiKey) == "":
		return nil, errors.New("openai: API key is required")
	case strings.ContainsAny(cfg.apiKey, "\r\n"):
		return nil, errors.New("openai: API key contains a newline")
	case strings.TrimSpace(cfg.model) == "":
		return nil, errors.New("openai: model is required")
	case cfg.dimensions < 0:
		return nil, errors.New("openai: dimensions must be non-negative")
	case cfg.httpClient == nil:
		return nil, errors.New("openai: HTTP client is nil")
	}

	endpoint, err := embeddingsURL(cfg.endpoint)
	if err != nil {
		return nil, err
	}

	cfg.endpoint = endpoint
	return cfg, nil
}

// EmbedText returns a caller-owned embedding for one nonempty text value.
func (c *Client) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("openai: text is empty")
	}

	vectors, err := c.embed(ctx, text, 1, "embed text")
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedBatch returns caller-owned embeddings in input order using one request.
// Empty batches do no work. Errors return no partial results; requests are not retried.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return make([][]float32, 0), nil
	}
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("openai: text at index %d is empty", i)
		}
	}
	return c.embed(ctx, texts, len(texts), "embed batch")
}

func (c *Client) embed(ctx context.Context, input any, count int, action string) ([][]float32, error) {
	if c == nil || c.httpClient == nil || c.endpoint == "" || c.apiKey == "" || c.model == "" {
		return nil, errors.New("openai: client is not initialized")
	}
	if ctx == nil {
		return nil, errors.New("openai: context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("openai: %s: %w", action, err)
	}

	body, err := json.Marshal(embeddingRequest{
		Model:          c.model,
		Input:          input,
		EncodingFormat: "float",
		Dimensions:     c.dimensions,
	})
	if err != nil {
		return nil, fmt.Errorf("openai: %s: encode request: %w", action, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: %s: create request: %w", action, err)
	}
	req.Header = cloneHeaders(c.headers)
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: %s: request: %w", action, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		message, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("openai: %s: status %d: read response: %w", action, resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("openai: %s: status %d: %s", action, resp.StatusCode, strings.TrimSpace(string(message)))
	}

	var response embeddingResponse
	decoder := json.NewDecoder(resp.Body)
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("openai: %s: decode response: %w", action, err)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("openai: %s: decode response: trailing JSON", action)
		}
		return nil, fmt.Errorf("openai: %s: decode response: trailing data: %w", action, err)
	}

	vectors, err := response.vectors(count, c.dimensions)
	if err != nil {
		return nil, fmt.Errorf("openai: %s: invalid response: %w", action, err)
	}
	return vectors, nil
}

type embeddingRequest struct {
	Model          string `json:"model"`
	Input          any    `json:"input"`
	EncodingFormat string `json:"encoding_format"`
	Dimensions     int    `json:"dimensions,omitempty"`
}

type embeddingResponse struct {
	Data []embeddingData `json:"data"`
}

type embeddingData struct {
	Embedding json.RawMessage `json:"embedding"`
	Index     *int            `json:"index"`
}

func (r embeddingResponse) vectors(count, dimensions int) ([][]float32, error) {
	if len(r.Data) != count {
		return nil, fmt.Errorf("expected %d embeddings, got %d", count, len(r.Data))
	}

	vectors := make([][]float32, count)
	vectorSize := 0
	for i, item := range r.Data {
		if item.Index == nil {
			return nil, fmt.Errorf("embedding %d has no index", i)
		}
		index := *item.Index
		if index < 0 || index >= count {
			return nil, fmt.Errorf("embedding index %d is out of range", index)
		}
		if vectors[index] != nil {
			return nil, fmt.Errorf("embedding index %d is duplicated", index)
		}

		vector, err := decodeVector(item.Embedding, index)
		if err != nil {
			return nil, err
		}
		if vectorSize == 0 {
			vectorSize = len(vector)
		} else if len(vector) != vectorSize {
			return nil, fmt.Errorf("embedding index %d has dimension %d, expected %d", index, len(vector), vectorSize)
		}
		if dimensions > 0 && len(vector) != dimensions {
			return nil, fmt.Errorf("embedding index %d has dimension %d, expected %d", index, len(vector), dimensions)
		}

		vectors[index] = vector
	}
	return vectors, nil
}

func decodeVector(raw json.RawMessage, index int) ([]float32, error) {
	// encoding/json accepts null for numeric elements, so reject it explicitly.
	if bytes.Contains(raw, []byte("null")) {
		return nil, fmt.Errorf("embedding index %d contains null", index)
	}
	var vector []float32
	if err := json.Unmarshal(raw, &vector); err != nil {
		return nil, fmt.Errorf("embedding index %d: %w", index, err)
	}
	if len(vector) == 0 {
		return nil, fmt.Errorf("embedding index %d is empty", index)
	}
	nonzero := false
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("embedding index %d value %d is non-finite", index, i)
		}
		nonzero = nonzero || value != 0
	}
	if !nonzero {
		return nil, fmt.Errorf("embedding index %d is zero", index)
	}
	return vector, nil
}

func embeddingsURL(base string) (string, error) {
	u, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("openai: invalid base URL: %w", err)
	}
	if u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return "", errors.New("openai: base URL must be an absolute HTTP or HTTPS URL")
	}
	if u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", errors.New("openai: base URL cannot contain user info, query, or fragment")
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/embeddings"
	u.RawPath = ""
	return u.String(), nil
}

func cloneHeaders(input http.Header) http.Header {
	output := make(http.Header, len(input))
	for key, values := range input {
		if strings.EqualFold(key, "Authorization") || strings.EqualFold(key, "Content-Type") {
			continue
		}
		output[key] = append([]string(nil), values...)
	}
	return output
}
