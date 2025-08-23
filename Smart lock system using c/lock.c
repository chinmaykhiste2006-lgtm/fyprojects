#include <stdio.h>
#include <stdlib.h>
#include <windows.h> // For Beep(), MessageBox(), Console Manipulation
#include <string.h>

// Structure to store password and hint
struct lock {
    char password[50];
    char hint[50];
};

// Function to trigger alarm (Beep sound)
void trigger_alarm() {
    Beep(1000, 500); // Beep at 1000 Hz for 500ms
}

// Function to flash the screen like a warning LED
void flash_screen() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    for (int i = 0; i < 1; i++) { // Flash 3 times
        SetConsoleTextAttribute(hConsole, BACKGROUND_RED | FOREGROUND_RED | FOREGROUND_INTENSITY);
        printf("\n***** WARNING! INCORRECT PASSWORD *****\n");
        Sleep(300);
        SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        Sleep(300);
    }
}

// Function to show a warning pop-up after 5 wrong attempts
void show_popup() {
    MessageBox(0, "You have entered the wrong password 5 times!\nThe system is temporarily locked.", "Security Warning", MB_ICONWARNING | MB_OK);
}

int main() {
    int tries, num, choice, no, n0;
    char enterpassword[50], gethint[50], ch;
    const char gethint1[] = "hint";
    const char tryagain[] = "try";

    printf("Enter number of doors: ");
    scanf("%d", &no);

    struct lock door[no];

    // Taking password and hint input
    printf("Enter details (password and hint)\n");
    for (int i = 0; i < no; i++) {
        printf("Door %d\n", i + 1);
    again:
        printf("Password: ");
        scanf("%s", door[i].password);
        num = door[i].password[0];
        if (num > 90 || strlen(door[i].password) < 8) {
            printf("First letter should be capital and password should be at least 8 characters long.\n");
            goto again;
        }

        printf("Do you want to set a hint? (y/n): ");
        scanf(" %c", &ch);
        if (ch == 'y' || ch == 'Y') {
            printf("Hint: ");
            scanf("%s", door[i].hint);
        } else {
            strcpy(door[i].hint, "null");
        }
    }

    // Unlocking process
    do {
        tries = 1;
        printf("Enter door number: ");
        scanf("%d", &n0);

        if (n0 < 1 || n0 > no) {
            printf("Invalid door number!\n");
            continue;
        }

        do {
            printf("Enter your password: ");
            scanf("%s", enterpassword);

            if (strcmp(enterpassword, door[n0 - 1].password) == 0) {
                printf("Door is opening...\n");
                break;
            } else {
                trigger_alarm(); // Beep sound
                flash_screen(); // Flash warning screen

                if (tries == 5) {
                    printf("You have entered the wrong password 5 times. System will activate in 10 seconds...\n");

                    show_popup(); // Show pop-up warning

                    // Countdown
                    for (int k = 10; k > 0; k--) {
                        printf("Time remaining: %d seconds...\n", k);
                        Sleep(1000);
                    }
                    printf("The system is now active.\n");
                } else {
                    if (strcmp(door[n0 - 1].hint, "null") != 0) {
                        printf("Incorrect password. Type 'try' to retry or 'hint' to get a hint: ");
                        scanf("%s", gethint);
                        if (strcmp(gethint, gethint1) == 0) {
                            printf("Your hint is: %s\n", door[n0 - 1].hint);
                        }
                    } else {
                        printf("Incorrect password. Type 'try' to retry: ");
                        scanf("%s", gethint);
                    }
                }
                tries++;
            }
        } while (tries <= 5);

    } while (1);

    return 0;
}
