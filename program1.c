// C program to demonstrate 
// Flawfinder
#include <stdio.h>
#include <string.h>
  
// Driver code
int main()
{
    char temp[100];
    char str[] = "hello";
    strcpy(temp, str);
    printf("%s", temp);
    return 0;
}