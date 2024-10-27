//go:build !noasm && amd64
// AUTO-GENERATED BY GOCC -- DO NOT EDIT

TEXT ·f32_cosine_distance(SB), $0-32
	MOVQ x+0(FP), DI
	MOVQ y+8(FP), SI
	MOVQ result+16(FP), DX
	MOVQ size+24(FP), CX
	BYTE $0x55               // push	rbp
	WORD $0x8948; BYTE $0xe5 // mov	rbp, rsp
	LONG $0xe0e48348         // and	rsp, -32
	LONG $0x60ec8348         // sub	rsp, 96
	LONG $0xed57d0c5         // vxorps	xmm5, xmm5, xmm5
	LONG $0xe457d8c5         // vxorps	xmm4, xmm4, xmm4
	LONG $0xc957f0c5         // vxorps	xmm1, xmm1, xmm1
	LONG $0xdb57e0c5         // vxorps	xmm3, xmm3, xmm3
	WORD $0x8548; BYTE $0xc9 // test	rcx, rcx
	JE   LBB0_8
	LONG $0x20f98348         // cmp	rcx, 32
	JAE  LBB0_4
	LONG $0xd257e8c5         // vxorps	xmm2, xmm2, xmm2
	WORD $0x3145; BYTE $0xc0 // xor	r8d, r8d
	LONG $0xf657c8c5         // vxorps	xmm6, xmm6, xmm6
	LONG $0xc957f0c5         // vxorps	xmm1, xmm1, xmm1
	JMP  LBB0_3

LBB0_4:
	WORD $0x8949; BYTE $0xc8     // mov	r8, rcx
	LONG $0xe0e08349             // and	r8, -32
	LONG $0xc057f8c5             // vxorps	xmm0, xmm0, xmm0
	LONG $0x0429fcc5; BYTE $0x24 // vmovaps	ymmword ptr [rsp], ymm0
	WORD $0xc031                 // xor	eax, eax
	LONG $0xdb57e0c5             // vxorps	xmm3, xmm3, xmm3
	LONG $0xe457d8c5             // vxorps	xmm4, xmm4, xmm4
	LONG $0xed57d0c5             // vxorps	xmm5, xmm5, xmm5
	LONG $0xf657c8c5             // vxorps	xmm6, xmm6, xmm6
	LONG $0xff57c0c5             // vxorps	xmm7, xmm7, xmm7
	LONG $0x573841c4; BYTE $0xc0 // vxorps	xmm8, xmm8, xmm8
	LONG $0x573041c4; BYTE $0xc9 // vxorps	xmm9, xmm9, xmm9
	LONG $0xc957f0c5             // vxorps	xmm1, xmm1, xmm1
	LONG $0x572041c4; BYTE $0xdb // vxorps	xmm11, xmm11, xmm11
	LONG $0x571841c4; BYTE $0xe4 // vxorps	xmm12, xmm12, xmm12
	LONG $0x571041c4; BYTE $0xed // vxorps	xmm13, xmm13, xmm13

LBB0_5:
	LONG $0x6c29fcc5; WORD $0x2024 // vmovaps	ymmword ptr [rsp + 32], ymm5
	LONG $0x34107cc5; BYTE $0x87   // vmovups	ymm14, ymmword ptr [rdi + 4*rax]
	LONG $0x7c107cc5; WORD $0x2087 // vmovups	ymm15, ymmword ptr [rdi + 4*rax + 32]
	LONG $0x54107cc5; WORD $0x4087 // vmovups	ymm10, ymmword ptr [rdi + 4*rax + 64]
	LONG $0x4410fcc5; WORD $0x6087 // vmovups	ymm0, ymmword ptr [rdi + 4*rax + 96]
	LONG $0x1410fcc5; BYTE $0x86   // vmovups	ymm2, ymmword ptr [rsi + 4*rax]
	LONG $0xec28fcc5               // vmovaps	ymm5, ymm4
	LONG $0xe328fcc5               // vmovaps	ymm4, ymm3
	LONG $0x1c28fcc5; BYTE $0x24   // vmovaps	ymm3, ymmword ptr [rsp]
	LONG $0xb86dc2c4; BYTE $0xde   // vfmadd231ps	ymm3, ymm2, ymm14
	LONG $0x1c29fcc5; BYTE $0x24   // vmovaps	ymmword ptr [rsp], ymm3
	LONG $0xdc28fcc5               // vmovaps	ymm3, ymm4
	LONG $0xe528fcc5               // vmovaps	ymm4, ymm5
	LONG $0x6c28fcc5; WORD $0x2024 // vmovaps	ymm5, ymmword ptr [rsp + 32]
	LONG $0xb80dc2c4; BYTE $0xf6   // vfmadd231ps	ymm6, ymm14, ymm14
	LONG $0x74107cc5; WORD $0x2086 // vmovups	ymm14, ymmword ptr [rsi + 4*rax + 32]
	LONG $0xb80dc2c4; BYTE $0xdf   // vfmadd231ps	ymm3, ymm14, ymm15
	LONG $0xb805c2c4; BYTE $0xff   // vfmadd231ps	ymm7, ymm15, ymm15
	LONG $0x7c107cc5; WORD $0x4086 // vmovups	ymm15, ymmword ptr [rsi + 4*rax + 64]
	LONG $0xb805c2c4; BYTE $0xe2   // vfmadd231ps	ymm4, ymm15, ymm10
	LONG $0xb82d42c4; BYTE $0xc2   // vfmadd231ps	ymm8, ymm10, ymm10
	LONG $0x54107cc5; WORD $0x6086 // vmovups	ymm10, ymmword ptr [rsi + 4*rax + 96]
	LONG $0xb82de2c4; BYTE $0xe8   // vfmadd231ps	ymm5, ymm10, ymm0
	LONG $0xb87d62c4; BYTE $0xc8   // vfmadd231ps	ymm9, ymm0, ymm0
	LONG $0xb86de2c4; BYTE $0xca   // vfmadd231ps	ymm1, ymm2, ymm2
	LONG $0xb80d42c4; BYTE $0xde   // vfmadd231ps	ymm11, ymm14, ymm14
	LONG $0xb80542c4; BYTE $0xe7   // vfmadd231ps	ymm12, ymm15, ymm15
	LONG $0xb82d42c4; BYTE $0xea   // vfmadd231ps	ymm13, ymm10, ymm10
	LONG $0x20c08348               // add	rax, 32
	WORD $0x3949; BYTE $0xc0       // cmp	r8, rax
	JNE  LBB0_5
	LONG $0xc158a4c5               // vaddps	ymm0, ymm11, ymm1
	LONG $0xc0589cc5               // vaddps	ymm0, ymm12, ymm0
	LONG $0xc05894c5               // vaddps	ymm0, ymm13, ymm0
	LONG $0x197de3c4; WORD $0x01c1 // vextractf128	xmm1, ymm0, 1
	LONG $0xc158f8c5               // vaddps	xmm0, xmm0, xmm1
	LONG $0x0579e3c4; WORD $0x01c8 // vpermilpd	xmm1, xmm0, 1
	LONG $0xc158f8c5               // vaddps	xmm0, xmm0, xmm1
	LONG $0xc816fac5               // vmovshdup	xmm1, xmm0
	LONG $0xc958fac5               // vaddss	xmm1, xmm0, xmm1
	LONG $0xc658c4c5               // vaddps	ymm0, ymm7, ymm6
	LONG $0xc058bcc5               // vaddps	ymm0, ymm8, ymm0
	LONG $0xc058b4c5               // vaddps	ymm0, ymm9, ymm0
	LONG $0x197de3c4; WORD $0x01c2 // vextractf128	xmm2, ymm0, 1
	LONG $0xc258f8c5               // vaddps	xmm0, xmm0, xmm2
	LONG $0x0579e3c4; WORD $0x01d0 // vpermilpd	xmm2, xmm0, 1
	LONG $0xc258f8c5               // vaddps	xmm0, xmm0, xmm2
	LONG $0xd016fac5               // vmovshdup	xmm2, xmm0
	LONG $0xf258fac5               // vaddss	xmm6, xmm0, xmm2
	LONG $0x0458e4c5; BYTE $0x24   // vaddps	ymm0, ymm3, ymmword ptr [rsp]
	LONG $0xc058dcc5               // vaddps	ymm0, ymm4, ymm0
	LONG $0xc058d4c5               // vaddps	ymm0, ymm5, ymm0
	LONG $0x197de3c4; WORD $0x01c2 // vextractf128	xmm2, ymm0, 1
	LONG $0xc258f8c5               // vaddps	xmm0, xmm0, xmm2
	LONG $0x0579e3c4; WORD $0x01d0 // vpermilpd	xmm2, xmm0, 1
	LONG $0xc258f8c5               // vaddps	xmm0, xmm0, xmm2
	LONG $0xd016fac5               // vmovshdup	xmm2, xmm0
	LONG $0xd258fac5               // vaddss	xmm2, xmm0, xmm2
	WORD $0x3949; BYTE $0xc8       // cmp	r8, rcx
	LONG $0xe457d8c5               // vxorps	xmm4, xmm4, xmm4
	LONG $0xed57d0c5               // vxorps	xmm5, xmm5, xmm5
	JE   LBB0_7

LBB0_3:
	LONG $0x107aa1c4; WORD $0x8704 // vmovss	xmm0, dword ptr [rdi + 4*r8]
	LONG $0x107aa1c4; WORD $0x861c // vmovss	xmm3, dword ptr [rsi + 4*r8]
	LONG $0xb961e2c4; BYTE $0xd0   // vfmadd231ss	xmm2, xmm3, xmm0
	LONG $0xb979e2c4; BYTE $0xf0   // vfmadd231ss	xmm6, xmm0, xmm0
	LONG $0xb961e2c4; BYTE $0xcb   // vfmadd231ss	xmm1, xmm3, xmm3
	WORD $0xff49; BYTE $0xc0       // inc	r8
	WORD $0x394c; BYTE $0xc1       // cmp	rcx, r8
	JNE  LBB0_3

LBB0_7:
	LONG $0xd959cac5 // vmulss	xmm3, xmm6, xmm1
	LONG $0xca5aeac5 // vcvtss2sd	xmm1, xmm2, xmm2

LBB0_8:
	LONG $0xd351e2c5         // vsqrtss	xmm2, xmm3, xmm3
	LONG $0xd52ef8c5         // vucomiss	xmm2, xmm5
	JNE  LBB0_9
	LONG $0x2211fbc5         // vmovsd	qword ptr [rdx], xmm4
	WORD $0x8948; BYTE $0xec // mov	rsp, rbp
	BYTE $0x5d               // pop	rbp
	WORD $0xf8c5; BYTE $0x77 // vzeroupper
	BYTE $0xc3               // ret

LBB0_9:
	LONG $0xc25aeac5         // vcvtss2sd	xmm0, xmm2, xmm2
	LONG $0xe05ef3c5         // vdivsd	xmm4, xmm1, xmm0
	LONG $0x2211fbc5         // vmovsd	qword ptr [rdx], xmm4
	WORD $0x8948; BYTE $0xec // mov	rsp, rbp
	BYTE $0x5d               // pop	rbp
	WORD $0xf8c5; BYTE $0x77 // vzeroupper
	BYTE $0xc3               // ret
