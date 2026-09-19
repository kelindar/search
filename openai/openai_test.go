// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	cases := []struct {
		name    string
		apiKey  string
		model   string
		options []Option
	}{
		{name: "key", model: "model"},
		{name: "model", apiKey: "key"},
		{name: "url", apiKey: "key", model: "model", options: []Option{BaseURL("://bad")}},
		{name: "dimensions", apiKey: "key", model: "model", options: []Option{Dimensions(-1)}},
		{name: "client", apiKey: "key", model: "model", options: []Option{HTTPClient(nil)}},
		{name: "option", apiKey: "key", model: "model", options: []Option{nil}},
		{name: "newline", apiKey: "key\nvalue", model: "model"},
		{name: "relative url", apiKey: "key", model: "model", options: []Option{BaseURL("example.test/v1")}},
		{name: "scheme", apiKey: "key", model: "model", options: []Option{BaseURL("ftp://example.test/v1")}},
		{name: "user", apiKey: "key", model: "model", options: []Option{BaseURL("https://user@example.test/v1")}},
		{name: "query", apiKey: "key", model: "model", options: []Option{BaseURL("https://example.test/v1?x=1")}},
		{name: "fragment", apiKey: "key", model: "model", options: []Option{BaseURL("https://example.test/v1#part")}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			_, err := New(test.apiKey, test.model, test.options...)
			assert.Error(t, err)
		})
	}

	client, err := New("key", "model")
	require.NoError(t, err)
	assert.Equal(t, defaultBaseURL+"/embeddings", client.endpoint)
	assert.Equal(t, defaultTimeout, client.httpClient.Timeout)

	customHTTP := &http.Client{Timeout: time.Second}
	client, err = New("key", "openai/text-embedding-3-small",
		BaseURL("https://openrouter.ai/api/v1"), HTTPClient(customHTTP))
	require.NoError(t, err)
	assert.Equal(t, "https://openrouter.ai/api/v1/embeddings", client.endpoint)
	assert.Same(t, customHTTP, client.httpClient)
}

func TestRequests(t *testing.T) {
	type request struct {
		method      string
		path        string
		authorize   string
		contentType string
		testHeader  string
		body        []byte
	}
	var requests []request

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests = append(requests, request{
			method:      r.Method,
			path:        r.URL.Path,
			authorize:   r.Header.Get("Authorization"),
			contentType: r.Header.Get("Content-Type"),
			testHeader:  r.Header.Get("X-Test"),
			body:        body,
		})
		if len(requests) == 1 {
			_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[1,2]}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}]}`)
	}))
	defer server.Close()

	custom := http.Header{
		"Authorization": {"wrong"},
		"content-type":  {"text/plain"},
		"X-Test":        {"original"},
	}
	client, err := New("secret", "model", BaseURL(server.URL+"/v1/"), Dimensions(2), Headers(custom))
	require.NoError(t, err)
	custom.Set("X-Test", "changed")

	vector, err := client.EmbedText(context.Background(), "one")
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 2}, vector)

	vectors, err := client.EmbedBatch(context.Background(), []string{"one", "two"})
	require.NoError(t, err)
	assert.Equal(t, [][]float32{{1, 2}, {3, 4}}, vectors)
	require.Len(t, requests, 2)

	assert.Equal(t, http.MethodPost, requests[0].method)
	assert.Equal(t, "/v1/embeddings", requests[0].path)
	assert.Equal(t, "Bearer secret", requests[0].authorize)
	assert.Equal(t, "application/json", requests[0].contentType)
	assert.Equal(t, "original", requests[0].testHeader)

	var single struct {
		Model          string `json:"model"`
		Input          string `json:"input"`
		EncodingFormat string `json:"encoding_format"`
		Dimensions     *int   `json:"dimensions"`
	}
	require.NoError(t, json.Unmarshal(requests[0].body, &single))
	assert.Equal(t, "model", single.Model)
	assert.Equal(t, "one", single.Input)
	assert.Equal(t, "float", single.EncodingFormat)
	require.NotNil(t, single.Dimensions)
	assert.Equal(t, 2, *single.Dimensions)

	var batch struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	require.NoError(t, json.Unmarshal(requests[1].body, &batch))
	assert.Equal(t, "model", batch.Model)
	assert.Equal(t, []string{"one", "two"}, batch.Input)
}

func TestEmpty(t *testing.T) {
	transport := &countTransport{}
	client, err := New("key", "model", BaseURL("http://example.test/v1"), HTTPClient(&http.Client{Transport: transport}))
	require.NoError(t, err)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	empty, err := client.EmbedBatch(canceled, nil)
	require.NoError(t, err)
	assert.Empty(t, empty)
	assert.NotNil(t, empty)

	_, err = client.EmbedText(context.Background(), "")
	assert.Error(t, err)
	_, err = client.EmbedBatch(context.Background(), []string{"one", ""})
	assert.Error(t, err)
	_, err = client.EmbedBatch(canceled, []string{"one"})
	assert.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, 0, transport.count)
}

func TestUninitialized(t *testing.T) {
	var nilClient *Client
	_, err := nilClient.EmbedText(context.Background(), "one")
	assert.ErrorContains(t, err, "client is not initialized")

	client, err := New("key", "model")
	require.NoError(t, err)
	_, err = client.EmbedText(nil, "one")
	assert.ErrorContains(t, err, "context is nil")
}

func TestCancel(t *testing.T) {
	started := make(chan struct{})
	finished := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
		close(finished)
	}))
	defer server.Close()

	client, err := New("key", "model", BaseURL(server.URL))
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errorsCh := make(chan error, 1)
	go func() {
		_, callErr := client.EmbedText(ctx, "one")
		errorsCh <- callErr
	}()
	<-started
	cancel()

	assert.ErrorIs(t, <-errorsCh, context.Canceled)
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("server did not observe cancellation")
	}
}

func TestErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	client, err := New("key", "model", BaseURL(server.URL))
	require.NoError(t, err)
	_, err = client.EmbedText(context.Background(), "one")
	assert.ErrorContains(t, err, "status 400")
	assert.ErrorContains(t, err, "bad request")
	server.Close()

	transport := &errorTransport{err: errors.New("transport failed")}
	client, err = New("key", "model", HTTPClient(&http.Client{Transport: transport}))
	require.NoError(t, err)
	_, err = client.EmbedText(context.Background(), "one")
	assert.ErrorIs(t, err, transport.err)
}

func TestMalformed(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		batch      bool
		dimensions int
	}{
		{name: "empty", body: `{"data":[]}`},
		{name: "missing-index", body: `{"data":[{"embedding":[1]}]}`},
		{name: "out-of-range", body: `{"data":[{"index":1,"embedding":[1]}]}`},
		{name: "empty-vector", body: `{"data":[{"index":0,"embedding":[]}]}`},
		{name: "zero-vector", body: `{"data":[{"index":0,"embedding":[0,0]}]}`},
		{name: "nonfinite", body: `{"data":[{"index":0,"embedding":[NaN]}]}`},
		{name: "overflow", body: `{"data":[{"index":0,"embedding":[1e40]}]}`},
		{name: "null", body: `{"data":[{"index":0,"embedding":[null,1]}]}`},
		{name: "dimension", body: `{"data":[{"index":0,"embedding":[1,2]}]}`, dimensions: 3},
		{name: "duplicate", body: `{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}`, batch: true},
		{name: "inconsistent", body: `{"data":[{"index":0,"embedding":[1,2]},{"index":1,"embedding":[3]}]}`, batch: true},
		{name: "decode", body: `{`},
		{name: "trailing", body: `{"data":[{"index":0,"embedding":[1]}]} {}`, batch: false},
		{name: "trailing data", body: `{"data":[{"index":0,"embedding":[1]}]} trailing`, batch: false},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.WriteString(w, test.body)
			}))
			defer server.Close()
			client, err := New("key", "model", BaseURL(server.URL), Dimensions(test.dimensions))
			require.NoError(t, err)

			if test.batch {
				vectors, callErr := client.EmbedBatch(context.Background(), []string{"one", "two"})
				assert.Error(t, callErr)
				assert.Nil(t, vectors)
				return
			}
			vector, callErr := client.EmbedText(context.Background(), "one")
			assert.Error(t, callErr)
			assert.Nil(t, vector)
		})
	}
}

func TestResponseReadError(t *testing.T) {
	client, err := New("key", "model", HTTPClient(&http.Client{
		Transport: &responseTransport{response: &http.Response{
			StatusCode: http.StatusBadGateway,
			Body:       &readErrorBody{},
		}},
	}))
	require.NoError(t, err)

	_, err = client.EmbedText(context.Background(), "one")
	assert.ErrorContains(t, err, "read response")
}

type countTransport struct {
	count int
}

func (t *countTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.count++
	return nil, errors.New("unexpected request")
}

type errorTransport struct {
	err error
}

func (t *errorTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, t.err
}

type responseTransport struct {
	response *http.Response
}

func (t *responseTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return t.response, nil
}

type readErrorBody struct{}

func (*readErrorBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func (*readErrorBody) Close() error { return nil }

var _ http.RoundTripper = (*countTransport)(nil)
var _ http.RoundTripper = (*errorTransport)(nil)
var _ http.RoundTripper = (*responseTransport)(nil)
