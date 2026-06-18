#include <stdio.h>
#include <signal.h>

void exit_handler(int n_signal) {
    printf("\nCODE %d\n", n_signal);
    printf("Exiting ...\n");
}

int main(int argc, char *argv[]) {
    signal(SIGINT, &exit_handler);
    getchar();
}