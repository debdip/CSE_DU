.global your_asm_fn



.section .text


your_asm_fn:
push $0xc105ed
mov %esp,%eax

push $.str


push %eax

push $0xacce55ed


call practice
leave
ret

.section .data
.str:
  .asciz "coolfood"



