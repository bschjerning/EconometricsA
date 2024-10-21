clear all
cd "C:\Users\WGK649\Dropbox\Economics\Undervisning\Econometrics A\2022\Hjemmeopgaver\A3"

cap log close
cap log using "X2017S_log.log", text replace

/*** LOAD DATA ***/
use X2017S_MetricsI

/*** PROBLEM 1.1 ***/
summarize dsL dIPWusch dIPWusmx college foreignborn routine
quietly estpost summarize dsL dIPWusch dIPWusmx college foreignborn routine, de listwise
esttab using T1.tex, replace cells("mean(fmt(%4.2f)) p50 min max") nomtitle nonumber title(Summary Statistics, 1990-2007)

summarize dsL dIPWusch dIPWusmx college foreignborn routine if t2==0
quietly estpost summarize dsL dIPWusch dIPWusmx college foreignborn routine if t2==0, de listwise
esttab using T2.tex, replace cells("mean(fmt(%4.2f)) p50 min max") nomtitle nonumber title(Summary Statistics, 1990-2000)

summarize dsL dIPWusch dIPWusmx college foreignborn routine if t2==1
quietly estpost summarize dsL dIPWusch dIPWusmx college foreignborn routine if t2==1, de listwise
esttab using T3.tex, replace cells("mean(fmt(%4.2f)) p50 min max") nomtitle nonumber title(Summary Statistics, 2000-2007)

/*** PROBLEM 1.2 ***/
regress dsL t2 dIPWusch college foreign routine, robust 
eststo ols

/*** PROBLEM 2.1 ***/
regress dIPWusch t2 dIPWotch college foreign routine, robust
eststo first1

/*** PROBLEM 2.2 ***/
ivregress 2sls dsL (dIPWusch=dIPWotch) t2 college foreign routine, robust
eststo iv

/*** PROBLEM 2.3 ***/
ivregress 2sls dsL (dIPWusch=dIPWotch dIPWukch) t2 college foreign routine, robust
eststo ivoi
predict IVres, residual
estat overid

reg IVres dIPWotch dIPWukch t2 college foreign routine, robust
display "OI TEST =" e(N)*e(r2)

reg dIPWusch dIPWotch dIPWukch t2 college foreign routine
eststo first2
test dIPWotch dIPWukch

/*** PROBLEM 2.4 ***/
ivregress 2sls dsL (dIPWusch dIPWusmx=dIPWotch dIPWotmx) t2 college foreign routine, robust
eststo ivmex
test dIPWusch=dIPWusmx

regress dIPWusch dIPWotch dIPWotmx t2 college foreign routine, robust
eststo first3

regress dIPWusmx dIPWotch dIPWotmx t2 college foreign routine, robust
eststo first4

esttab ols iv ivoi ivmex using T4.tex, replace title(OLS and IV results) mtitles("OLS (Prob 1.2)" "IV (Prob 2.2)" "IV (Prob 2.3)" "IV (Prob 2.4)") se star(* 0.05)
esttab first1 first2 first3 first4 using T5.tex, replace fragment title(First stage regressions (Problem 2)) se star(* 0.05)

/*** PROBLEM 3.1 ***/
gen t2dIPWusch=t2*dIPWusch
gen t2dIPWotch=t2*dIPWotch

regress dIPWusch dIPWotch t2dIPWotch t2 college foreign routine, robust
eststo first5
predict res1, residual 
predict dIPWuschhat, xb
test dIPWotch t2dIPWotch

regress t2dIPWusch dIPWotch t2dIPWotch t2 college foreign routine, robust
eststo first6
predict res2, residual
predict t2dIPWuschhat, xb
test dIPWotch t2dIPWotch

regress dsL dIPWuschhat t2dIPWuschhat t2 college foreign routine, robust
eststo iv2sls

/*** PROBLEM 3.2 ***/
ivregress 2sls dsL (dIPWusch t2dIPWusch=dIPWotch t2dIPWotch) t2 college foreign routine, robust
test dIPWusch=t2dIPWusch
eststo ivt2

/*** PROBLEM 3.3 ***/
regress dsL dIPWusch t2dIPWusch t2 college foreign routine res1 res2, robust
test res1 res2

esttab first5 first6 iv2sls ivt2 using T6.tex, replace se star(* 0.05) fragment title(Problem 3) mtitles("1st" "1st" "2nd" "IV")

/*** PROBLEM 5: SIMULATION EXPERIMENT ***/
clear all

*DEFINE PROGRAM THAT SPECIFIES THE DGP
program dgp, rclass
	drop _all
	
	*SET NUMBER OF CURRENT OBSERVATIONS 
	set obs 1000
	
	*DATA GENERATING PROCESS
	gen xstar=rnormal(1,2)
	gen e1=rnormal(0,1)
	gen x=xstar+e1
	
	gen e2=$rho *e1+rnormal(0,1)
	gen z=$theta *xstar+e2
	
	gen y=4+3*xstar+rnormal(0,1)
		
	*CALCULATE OLS ESTIMATES
	regress y x
	return scalar b_ols=_b[x]
	
	*CALCULATE IV ESTIMATES
	ivregress 2sls y (x=z)
	return scalar b_iv=_b[x]
end

*MONTE CARLO SIMULATION 1
global rho 0.5
global theta 1 
simulate b_ols=r(b_ols) b_iv=r(b_iv), seed(1479) reps(500) nodots:dgp
estpost tabstat b_ols b_iv, stat(mean sd min max) columns(statistics) 
esttab . using MC1, replace tex cells("mean(fmt(%6.3f)) sd min max") noobs nonumber 

*MONTE CARLO SIMULATION 2
global rho 0
global theta 1 

simulate b_ols=r(b_ols) b_iv=r(b_iv), seed(1479) reps(500) nodots:dgp
estpost tabstat b_iv, stat(mean sd min max) columns(statistics) 
esttab . using MC2, replace tex cells("mean(fmt(%6.3f)) sd min max") noobs nonumber 

*MONTE CARLO SIMULATION 3
global rho 1
global theta 1 
simulate b_ols=r(b_ols) b_iv=r(b_iv), seed(1479) reps(500) nodots:dgp
estpost tabstat b_iv, stat(mean sd min max) columns(statistics)  
esttab . using MC3, replace tex cells("mean(fmt(%6.3f)) sd min max") noobs nonumber 

*MONTE CARLO SIMULATION 4
global rho -0.5
global theta 1 
simulate b_ols=r(b_ols) b_iv=r(b_iv), seed(1479) reps(500) nodots:dgp
estpost tabstat b_iv, stat(mean sd min max) columns(statistics)   
esttab . using MC4, replace tex cells("mean(fmt(%6.3f)) sd min max") noobs nonumber 

cap log close
