/*
 * forward/backward procedure 
 * 
 * to be used for HMM models.
 * 
 * Jean-Louis Durrieu, 2012
 * 
 */

#include <math.h>
#include <stdlib.h>

/* defining some global constants */
#define REALMIN 10E-9
#define REALMAX 10E9
#define MINUS_INFINITY -1000000000

/* some useful macros */
#define max(x,y) x<y?y:x
#define min(x,y) x<y?x:y
#define round(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

/* Core functions */
void computeLogDensity(float *pX, int F, int N, 
                       double *pMu, double *pSigma, 
		       int K, 
		       double *plogdens, double *pdens);
void computeForwardBackward(int N, int S, double *pi, double *A, double *dens, double *logdens, double* logAlpha, double *logBeta, double* alpha, double *beta);
void computeViterbiPath(int N, int S, double *pi, double *A, double *logdens, double *indBestPath);

/* instrumental functions */
void maxind(double * segmentToAnalyze, int length, double *maximum, double *index);

void computeLogDensity(double *pX, int F, int N, 
                       double *pMu, double *pSigma, 
		       int K, 
		       double *plogdens, double *pdens)
{
  double SXku;
  int n,k,f;
  /* Forward iterations: (computing alpha and the densities) */
  for(n=0;n<N;n++)
    {
      /* first compute the loglikelihoods... */
      for(k=0;k<K;k++)
        {
	  plogdens[k+K*n] = - F * 0.5 * log((double) 6.28318530717959);
	}
      for(f=0;f<F;f++)
        {
	  //printf("%5f ", pX[f+F*n]);
	  for(k=0;k<K;k++)
            {
	      //SXku = pSigma[f+F*k];
	      SXku = max(pSigma[f+F*k], REALMIN);
	      plogdens[k+K*n] = plogdens[k+K*n] - 0.5 * log(SXku) - 0.5 * (pX[f+F*n] - pMu[f+F*k]) * (pX[f+F*n] - pMu[f+F*k]) / SXku;
            }
        }
      
      //printf("\n");
      
      for(k=0;k<K;k++)
        {
	  pdens[k+K*n] = exp(plogdens[k+K*n]);
        }
    }
/*   for(f=0;f<F;f++) */
/*     { */
/*       for(k=0;k<K;k++) */
/* 	{ */
/* 	  printf("%10f ", pMu[f+F*k]); */
/* 	} */
/*       printf("\n"); */
/*     } */
}

/*
 * COMPUTING THE FORWARD BACKWARD PROCEDURE
 * WITH A VERY SPECIFIC SCALING SCHEME
 * ONLY SUITABLE WHEN _NOT_ UPDATING
 * THE TRANSITION PROBABILITIES
 */
void computeForwardBackward(int N, int S, double *pi, 
			    double *A, double *dens, double *logdens, 
			    double* logAlpha, double *logBeta, 
			    double* alpha, double *beta)
{
    int n,s,t,m;
    
    double maxLogAlpha=MINUS_INFINITY, maxAlpha=MINUS_INFINITY; 
    double maxBeta=MINUS_INFINITY, maxLogBeta=MINUS_INFINITY; 
    double *tmpAlpha, *tmpLogBeta;
    
    tmpAlpha = (double*) malloc(S*N*sizeof(double));
    tmpLogBeta = (double*) malloc(S*S*sizeof(double));
    
    /* Initializing */
    for(s=0;s<S;s++)
    {
        logAlpha[s] = log(pi[s]) + logdens[s];
        if (logAlpha[s]>maxLogAlpha)
            maxLogAlpha = logAlpha[s];
        
        logBeta[s+S*(N-1)] = 0.0;
        beta[s+S*(N-1)] = 1.0;
    }
    for(s=0;s<S;s++)
    {
        logAlpha[s] = logAlpha[s] - maxLogAlpha;
    }
    
    for(s=0;s<S;s++)
    {
        alpha[s] = exp(logAlpha[s]);
    }
    

    for(n=1;n<N;n++)
    {
        /* FORWARD variables: */
        maxLogAlpha=MINUS_INFINITY;
        for(s=0;s<S;s++)
        {
            logAlpha[s+S*n] = logdens[s+S*n];
	    // printf("logdens s, n:%f, %d, %d\n", logdens[s+S*n], s, n);
            //if (logAlpha[s+S*n]>maxLogAlpha)
            //    maxLogAlpha = logAlpha[s+S*n];
        }
        //for(s=0;s<S;s++)
        //{
        //    logAlpha[s+S*n] = logAlpha[s+S*n] - maxLogAlpha;
        //}
        maxAlpha=MINUS_INFINITY;
        //maxLogAlpha=MINUS_INFINITY;
        for(s=0;s<S;s++)
        { 
            tmpAlpha[s+S*n] = 0.0;
            for(t=0;t<S;t++)
            {
                tmpAlpha[s+S*n] = tmpAlpha[s+S*n] + alpha[t+S*(n-1)] * A[t+s*S];
            }
            if (log(tmpAlpha[s+S*n])>maxAlpha)
                maxAlpha = log(tmpAlpha[s+S*n]);
        }
        for(s=0;s<S;s++)
        {
            logAlpha[s+S*n] = logAlpha[s+S*n] + log(tmpAlpha[s+S*n]) - maxAlpha;
            if (logAlpha[s+S*n]>maxLogAlpha)
                maxLogAlpha = logAlpha[s+S*n];
        }
        for(s=0;s<S;s++)
        {
            logAlpha[s+S*n] = logAlpha[s+S*n] - maxLogAlpha;
            alpha[s+S*n] = exp(logAlpha[s+S*n]);
	    // DEBUG
	    //printf("logalpha s, n:%f, %d, %d\n", logAlpha[s+S*n], s, n);
	    //printf("alpha s, n:%f, %d, %d\n", alpha[s+S*n], s, n);
        }
    
        /* BACKWARD variables: */
        maxBeta=1; /* should prevent the algo to divide by 0... */
        m = N-n-1;
        
        maxLogBeta=MINUS_INFINITY;
        
        for(s=0;s<S;s++)
        {
            for(t=0;t<S;t++)
            {
                tmpLogBeta[t+S*s] = logBeta[t+S*(m+1)] + log(A[t+s*S]) + logdens[t+S*(m+1)];
                if (maxLogBeta<tmpLogBeta[t+S*s])
                    maxLogBeta = tmpLogBeta[t+S*s];
            }
        }
        for(s=0;s<S;s++)
        {
            beta[s+S*m] = 0.0;
            for(t=0;t<S;t++)
            {
                beta[s+S*m] = beta[s+S*m] + exp(tmpLogBeta[t+S*s] - maxLogBeta);
            }
            if (beta[s+S*m]>maxBeta)
                maxBeta = beta[s+S*m];
        }
        
        for(s=0;s<S;s++)
        {
            beta[s+S*m] = beta[s+S*m] / maxBeta;
            logBeta[s+S*m] = log(beta[s+S*m]);
        }
        
    }
    
    free(tmpAlpha);
    free(tmpLogBeta);
}

void computeViterbiPath(int N, int S, double *pi, double *A, double *logdens, double *indBestPath)
{
  int n, s, s_;
  double tmp;
  double *pathLogLikelihood;
  int *indexAntecedent;
  
  indexAntecedent = (int*) malloc(N*S*sizeof(int));
  pathLogLikelihood = (double*) malloc(N*S*sizeof(double));
  
  for(s=0;s<S;s++)
    {
      pathLogLikelihood[s+0*S] = logdens[s+0*S] + log(pi[s]);
      indexAntecedent[s+0*S] = 0;
    }
  
  for(n=1;n<N;n++)
    {
      for(s=0;s<S;s++)
        {
	  pathLogLikelihood[s+n*S] = pathLogLikelihood[0+(n-1)*S] + log(A[s+0*S]);
	  indexAntecedent[s+n*S] = 0;
	  for(s_=1;s_<S;s_++)
            {
	      tmp = pathLogLikelihood[s_+(n-1)*S] + log(A[s+s_*S]); 
	      if(tmp>pathLogLikelihood[s+n*S])
                {
		  pathLogLikelihood[s+n*S] = tmp;
		  indexAntecedent[s+n*S] = s_;
                }
            }
	  pathLogLikelihood[s+n*S]= pathLogLikelihood[s+n*S] + logdens[s+n*S];
        }
    }
  maxind(&pathLogLikelihood[(N-1)*S], S, &tmp, &indBestPath[N-1]);
  for(n=N-2;n>-1;n--)
    {
      indBestPath[n] = indexAntecedent[(n+1)*S+(int)indBestPath[n+1]];
    }
  
  free(pathLogLikelihood);free(indexAntecedent);
}

void maxind(double * segmentToAnalyze, int length, double *maximum, double *index)
{
  int i=0;
    
  /* initialization:*/
  *maximum = segmentToAnalyze[0];
  *index = 0;
  for(i=1;i<length;i++)
    {
      if(segmentToAnalyze[i]>*maximum)
        {
	  *maximum = segmentToAnalyze[i];
	  *index = 1.0*i;
        }
    }
}
