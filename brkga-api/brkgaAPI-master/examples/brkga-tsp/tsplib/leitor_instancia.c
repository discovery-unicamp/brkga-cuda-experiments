#include <stdio.h>

int return_dimension(char *s);
int is_digit(char c);

int main(){
	FILE *f = fopen("lu980.tsp.txt", "r");
	char st[1000];

//	fscanf(f, "%s", st);
	fgets(st, 1000, f); //read name line
	fgets(st, 1000, f); //read comment line
	fgets(st, 1000, f); //read type line
	fgets(st, 1000, f); //read dimension line
	int dimension = return_dimension(st);
	fgets(st, 1000, f); //read edgen type line
	fgets(st, 1000, f); //read node section line

	int aux; double x,y;

	for(int i=0; i<dimension; i++){
		fscanf(f, "%d %lf %lf", &aux, &x, &y);
		printf("%d %lf %lf\n",aux,x,y);
	}

	fclose(f);


}


int return_dimension(char *s){
	int i=0;
	while (s[i]!='\0' && !is_digit(s[i]))
		i++;
	int result=0;
	while(is_digit(s[i])){
		result = result*10 + (s[i]-48);
		i++;
	}
	return result;

}

int is_digit(char c){
	if(c>=48 && c<=57)
		return 1;
	return 0;
}