.section .data
file_to_run:
.asciz       "/bin/sh"

.section .text
.globl main

main:
    pushl %ebp
    movl %esp, %ebp
    subl $0x8, %esp         

    movl $file_to_run, %edi
    movl %edi, -0x8(%ebp)   
    movl $0, -0x4(%ebp)

    movl $11, %eax                      # sys_execve
    movl $file_to_run, %ebx              # expected file to execute       
    leal -8(%ebp), %ecx                 # parameters
    movl $0, %edx                       # environment block
    int  $0x80              

    leave
    ret
