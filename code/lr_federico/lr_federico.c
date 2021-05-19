/* by FRT */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define NDEBUG
#include <assert.h>

#define K 3
#define MAX_DEG 3

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((_ira[_ip++] = _ira[_ip1++] + _ira[_ip2++]) ^ _ira[_ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)

struct varNode {
  int deg, whereis, rearrSize;
  struct checkNode * p2c[MAX_DEG];
  struct varNode * p2d[K-1];
} *v;

struct checkNode{
  int isOff;
  struct varNode * p2v[K];
} *c;

int N, N1, N2, N3, M, size[MAX_DEG+1], *list;
struct varNode ** class[MAX_DEG+1];

/* variabili globali per il generatore random */
unsigned int myrand, _ira[256];
unsigned char _ip, _ip1, _ip2, _ip3;

unsigned int randForInit(void) {
  unsigned long long int y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7FFFFFFF) + (y >> 31);
  myrand = (myrand & 0x7FFFFFFF) + (myrand >> 31);
  return myrand;
}

void initRandom(void) {
  int i;
  
  _ip = 128;
  _ip1 = _ip - 24;    
  _ip2 = _ip - 55;    
  _ip3 = _ip - 61;
  
  for (i = _ip3; i < _ip; i++) {
    _ira[i] = randForInit();
  }
}

float gaussRan(void) {
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;

  if (iset == 0) {
    do {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}

void error(char *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

void allocateMem(void) {
  int i;

  v = (struct varNode *)calloc(N, sizeof(struct varNode));
  c = (struct checkNode *)calloc(M, sizeof(struct checkNode));
  list = (int*)calloc(K*M, sizeof(int));
  for (i = 0; i <= MAX_DEG; i++)
    class[i] = (struct varNode **)calloc(N, sizeof(struct varNode *));
}

void makeGraph(void) {
  int i, j, k, num, index;
  FILE *fptr;
  fptr = fopen("./graph.txt","w");

  num = 0;
  for (i = 0; i < N1; i++)
    list[num++] = i;
  for (; i < N1+N2; i++) {
    list[num++] = i;
    list[num++] = i;
  }  
  for (; i < N; i++) {
    list[num++] = i;
    list[num++] = i;
    list[num++] = i;
  }  
  assert(num == K*M);
  // following lines are valid only for K=3
  while (num) {
    index = (int)(num * FRANDOM);
    i = list[index];
    list[index] = list[--num];
    list[num] = i;
    do {
      index = (int)(num * FRANDOM);
      j = list[index];
    } while (j == i);
    list[index] = list[--num];
    list[num] = j;
    do {
      index = (int)(num * FRANDOM);
      k = list[index];
    } while (k == i || k == j);
    list[index] = list[--num];
    list[num] = k;
  }
  for (i = 0; i < N; i++) {
    v[i].deg = 0;
    v[i].rearrSize = 0;
  }
  for (i = 0; i < M; i++) {
    c[i].isOff = 0;
    fprintf(fptr, "%d ", i+1);
    for (j = 0; j < K; j++) {
      k = list[K*i+j];
      c[i].p2v[j] = v + k;
      fprintf(fptr, "%d ", k+1);
      v[k].p2c[v[k].deg] = c + i;
      v[k].deg++;
    }
    fprintf(fptr, "\n");
  }
  for (i = 0; i <= MAX_DEG; i++)
    size[i] = 0;
  for (i = 0; i < N; i++) {
    v[i].whereis = size[v[i].deg]++;
    class[v[i].deg][v[i].whereis] = v + i;
  }
  fclose(fptr);
}

void lower(struct varNode * pv) {
  int i;

  assert(pv->deg);
  assert(pv->whereis >= 0);
  class[pv->deg][pv->whereis] = class[pv->deg][--size[pv->deg]];
  class[pv->deg][pv->whereis]->whereis = pv->whereis;
  for (i = 0; i < pv->deg; i++)
    if (pv->p2c[i]->isOff)
      pv->p2c[i] = pv->p2c[--(pv->deg)];
  pv->whereis = size[pv->deg]++;
  class[pv->deg][pv->whereis] = pv;
}

void checkClasses(void) {
  int i, j, k;

  for (j = 0; j <= MAX_DEG; j++)
    for (i = 0; i < size[j]; i++) {
      assert(class[j][i]->deg == j);
      assert(class[j][i]->whereis == i);
      for (k = 0; k < class[j][i]->deg; k++)
	assert(!class[j][i]->p2c[k]->isOff);
    }
}

void leafRemoval(void) {
  int index, i, j;
  struct varNode * pv;
  struct checkNode * pc;
  
  while (size[1]) {
    index = (int)(FRANDOM * size[1]);
    pv = class[1][index];
    class[1][index] = class[1][--size[1]];
    class[1][index]->whereis = index;
    pv->whereis = -1;
    assert(pv->deg == 1);
    pc = pv->p2c[0];
    assert(!pc->isOff);
    pc->isOff = 1;
    i = 0;
    for (j = 0; j < K; j++)
      if (pc->p2v[j] != pv)
	pv->p2d[i++] = pc->p2v[j];
    assert(i == K-1);

    for (i = 0; i < K-1; i++)
      lower(pv->p2d[i]);
    //checkClasses();
  }
}

int numDes(struct varNode * pv) {
  if (pv->deg) {
    return numDes(pv->p2d[0]) + numDes(pv->p2d[1]);
   } else {
    pv->rearrSize++;
    return 1;
  }
}


int main(int argc, char *argv[]) {
  int i, *hW, *vW;
  double f1, f2;
  FILE *fptr = fopen("weights.txt", "w");
  FILE *devran = fopen("random.txt","r"); // edited by Ste
  fread(&myrand, 4, 1, devran);
  fclose(devran);

  if (argc != 4) {
    fprintf(stderr, "usage: %s N f1 f2\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  N = atoi(argv[1]);
  f1 = atof(argv[2]);
  f2 = atof(argv[3]);
  initRandom();

  N1 = (int)(f1 * N + 0.5);
  N2 = (int)(f2 * N + 0.5);
  N3 = N - N1 - N2;
  if (N3 < 0) error("N3 negative");

  if ((N1 + 2 * N2) % 3 == 1) {
    if (N2) {
      N1++;
      N2--;
    } else {
      N1 -= 2;
      N2 += 2;
    }
  }  
  if ((N1 + 2 * N2) % 3 == 2) {
    if (N1) {
      N1--;
      N2++;
    } else {
      N1 += 2;
      N2 -= 2;
    }
  }
  if (N1 < 0) error("N1 negative");
  if (N2 < 0) error("N2 negative");
  if ((N1 + 2 * N2) % 3) error("non divisibile per 3");
  if (N1 + N2 + N3 != N) error("la somma non torna");
  M = (N1 + 2 * N2 + 3 * N3) / 3;
  
  allocateMem();
  makeGraph();
  //checkClasses();
  leafRemoval();
  if (size[1] + size[2] + size[3])
    printf("%i %i %i %i\n", size[0], size[1], size[2], size[3]);
  else {
    for (i = 0; i < N; i++)
      assert((v[i].deg == 0 && v[i].whereis >= 0) || (v[i].deg == 1 && v[i].whereis == -1));
    printf("# rate = %g\n", (double)size[0]/N);
    // printf("# 1:size  2:hW       3:vW   4:hWnorm   5:vWnorm\n");
    hW = (int*)calloc(N, sizeof(int));
    vW = (int*)calloc(N, sizeof(int));
    for (i = 0; i < N; i++)
      if (v[i].deg)
	hW[numDes(v+i)]++;
    for (i = 0; i < N; i++)
      if (v[i].deg == 0)
	vW[v[i].rearrSize]++;
    for (i = 0; i < N; i++){
      if (hW[i] || vW[i]){
	      // printf("%3i %10i %10i %10g %10g\n", i, hW[i], vW[i], (double)hW[i]/(N-size[0]),  (double)vW[i]/size[0]);
        fprintf(fptr, "%3i %10g %10g\n", i, (double)hW[i]/(N-size[0]),  (double)vW[i]/size[0]);
      }
    }
      printf("\n");
  fclose(fptr);


  }
  return EXIT_SUCCESS;
}