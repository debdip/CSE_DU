.global your_asm_fn



.section .text


your_asm_fn:
push $0x12341234

push $0xc001f00d
mov %esp,%eax
push %eax
push $0xc105ed
push $0xacce55ed

call practice
leave
ret

