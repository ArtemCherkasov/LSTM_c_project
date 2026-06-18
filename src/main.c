#include <stdio.h>
#include <signal.h>

void exit_handler(int n_signal) {
    printf("\nExiting ...\n");
}

int main(int argc, char *argv[]) {
    signal(SIGTERM, &exit_handler);
    getchar();
    return 0;
}