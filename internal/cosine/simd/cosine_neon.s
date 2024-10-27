//go:build !noasm && !darwin && arm64
// AUTO-GENERATED BY GOCC -- DO NOT EDIT

TEXT ·f32_cosine_distance(SB), $0-32
	MOVD x+0(FP), R0
	MOVD y+8(FP), R1
	MOVD result+16(FP), R2
	MOVD size+24(FP), R3
	WORD $0xa9bf7bfd       // stp	x29, x30, [sp, #-16]!
	WORD $0xf100107f       // cmp	x3, #4
	WORD $0x910003fd       // mov	x29, sp
	WORD $0x54000223       // b.lo	.LBB0_4
	WORD $0x6f00e401       // movi	v1.2d, #0000000000000000
	WORD $0x52800068       // mov	w8, #3
	WORD $0x6f00e402       // movi	v2.2d, #0000000000000000
	WORD $0xaa0103e9       // mov	x9, x1
	WORD $0x6f00e400       // movi	v0.2d, #0000000000000000
	WORD $0xaa0003ea       // mov	x10, x0

LBB0_2:
	WORD $0x3cc10543 // ldr	q3, [x10], #16
	WORD $0x3cc10524 // ldr	q4, [x9], #16
	WORD $0x91001108 // add	x8, x8, #4
	WORD $0x4e23cc62 // fmla	v2.4s, v3.4s, v3.4s
	WORD $0xeb03011f // cmp	x8, x3
	WORD $0x4e23cc81 // fmla	v1.4s, v4.4s, v3.4s
	WORD $0x4e24cc80 // fmla	v0.4s, v4.4s, v4.4s
	WORD $0x54ffff23 // b.lo	.LBB0_2
	WORD $0x927ef468 // and	x8, x3, #0xfffffffffffffffc
	WORD $0x14000005 // b	.LBB0_5

LBB0_4:
	WORD $0x6f00e400 // movi	v0.2d, #0000000000000000
	WORD $0xaa1f03e8 // mov	x8, xzr
	WORD $0x6f00e402 // movi	v2.2d, #0000000000000000
	WORD $0x6f00e401 // movi	v1.2d, #0000000000000000

LBB0_5:
	WORD $0x6e21d421 // faddp	v1.4s, v1.4s, v1.4s
	WORD $0xeb03011f // cmp	x8, x3
	WORD $0x6e22d442 // faddp	v2.4s, v2.4s, v2.4s
	WORD $0x6e20d403 // faddp	v3.4s, v0.4s, v0.4s
	WORD $0x7e30d820 // faddp	s0, v1.2s
	WORD $0x7e30d841 // faddp	s1, v2.2s
	WORD $0x7e30d862 // faddp	s2, v3.2s
	WORD $0x540006c2 // b.hs	.LBB0_12
	WORD $0xcb080069 // sub	x9, x3, x8
	WORD $0xf100213f // cmp	x9, #8
	WORD $0x54000503 // b.lo	.LBB0_10
	WORD $0x6f00e403 // movi	v3.2d, #0000000000000000
	WORD $0xd37ef50b // lsl	x11, x8, #2
	WORD $0x6f00e404 // movi	v4.2d, #0000000000000000
	WORD $0x927df12a // and	x10, x9, #0xfffffffffffffff8
	WORD $0x6f00e405 // movi	v5.2d, #0000000000000000
	WORD $0x9100416c // add	x12, x11, #16
	WORD $0x8b0a0108 // add	x8, x8, x10
	WORD $0x8b0c000b // add	x11, x0, x12
	WORD $0x6e040443 // mov	v3.s[0], v2.s[0]
	WORD $0x8b0c002c // add	x12, x1, x12
	WORD $0x6f00e402 // movi	v2.2d, #0000000000000000
	WORD $0xaa0a03ed // mov	x13, x10
	WORD $0x6e040424 // mov	v4.s[0], v1.s[0]
	WORD $0x6e040405 // mov	v5.s[0], v0.s[0]
	WORD $0x6f00e400 // movi	v0.2d, #0000000000000000
	WORD $0x6f00e401 // movi	v1.2d, #0000000000000000

LBB0_8:
	WORD $0xad7f9d66 // ldp	q6, q7, [x11, #-16]
	WORD $0xf10021ad // subs	x13, x13, #8
	WORD $0x9100816b // add	x11, x11, #32
	WORD $0x4e26ccc4 // fmla	v4.4s, v6.4s, v6.4s
	WORD $0xad7fc590 // ldp	q16, q17, [x12, #-16]
	WORD $0x4e27cce0 // fmla	v0.4s, v7.4s, v7.4s
	WORD $0x9100818c // add	x12, x12, #32
	WORD $0x4e26ce05 // fmla	v5.4s, v16.4s, v6.4s
	WORD $0x4e30ce03 // fmla	v3.4s, v16.4s, v16.4s
	WORD $0x4e27ce21 // fmla	v1.4s, v17.4s, v7.4s
	WORD $0x4e31ce22 // fmla	v2.4s, v17.4s, v17.4s
	WORD $0x54fffea1 // b.ne	.LBB0_8
	WORD $0x4e25d421 // fadd	v1.4s, v1.4s, v5.4s
	WORD $0xeb0a013f // cmp	x9, x10
	WORD $0x4e24d400 // fadd	v0.4s, v0.4s, v4.4s
	WORD $0x4e23d442 // fadd	v2.4s, v2.4s, v3.4s
	WORD $0x6e21d421 // faddp	v1.4s, v1.4s, v1.4s
	WORD $0x6e20d403 // faddp	v3.4s, v0.4s, v0.4s
	WORD $0x6e22d442 // faddp	v2.4s, v2.4s, v2.4s
	WORD $0x7e30d820 // faddp	s0, v1.2s
	WORD $0x7e30d861 // faddp	s1, v3.2s
	WORD $0x7e30d842 // faddp	s2, v2.2s
	WORD $0x54000180 // b.eq	.LBB0_12

LBB0_10:
	WORD $0xd37ef50a // lsl	x10, x8, #2
	WORD $0xcb080069 // sub	x9, x3, x8
	WORD $0x8b0a0028 // add	x8, x1, x10
	WORD $0x8b0a000a // add	x10, x0, x10

LBB0_11:
	WORD $0xbc404543 // ldr	s3, [x10], #4
	WORD $0xbc404504 // ldr	s4, [x8], #4
	WORD $0xf1000529 // subs	x9, x9, #1
	WORD $0x1f030461 // fmadd	s1, s3, s3, s1
	WORD $0x1f030080 // fmadd	s0, s4, s3, s0
	WORD $0x1f040882 // fmadd	s2, s4, s4, s2
	WORD $0x54ffff41 // b.ne	.LBB0_11

LBB0_12:
	WORD $0x1e210841 // fmul	s1, s2, s1
	WORD $0x1e21c022 // fsqrt	s2, s1
	WORD $0x2f00e401 // movi	d1, #0000000000000000
	WORD $0x1e202048 // fcmp	s2, #0.0
	WORD $0x54000081 // b.ne	.LBB0_14
	WORD $0xfd000041 // str	d1, [x2]
	WORD $0xa8c17bfd // ldp	x29, x30, [sp], #16
	WORD $0xd65f03c0 // ret

LBB0_14:
	WORD $0x1e22c000 // fcvt	d0, s0
	WORD $0x1e22c041 // fcvt	d1, s2
	WORD $0x1e611801 // fdiv	d1, d0, d1
	WORD $0xfd000041 // str	d1, [x2]
	WORD $0xa8c17bfd // ldp	x29, x30, [sp], #16
	WORD $0xd65f03c0 // ret