clear all
use "DNK 2018 Schools.dta", clear
keep CNT CNTSCHID PROAT5AB SC001Q01TA SC003Q01TA SC004Q01TA SC048Q02NA SC048Q03NA SCHLTYPE SCHSIZE 
drop if SCHSIZE == . 
drop if SC003Q01TA == .

label var PROAT5AB 		"Proportion of all teachers with relevant education"
label var SC048Q02NA 	"Percentage of students with special needs"
label var SC048Q03NA 	"Percentage of students from disadvantaged background"
label var SC001Q01TA 	"Community size"
label var SC003Q01TA 	"Average class size in reading"
label var SCHSIZE   	"Total number of students at school"
label var SC004Q01TA   	"Total number of students in 9th grade"

replace PROAT5AB   = 1 if PROAT5AB == .
replace SC048Q02NA = 0 if SC048Q02NA == .
replace SC048Q03NA = 0 if SC048Q03NA == .

gen     private = 0 
replace private = 1 if inlist(SCHLTYPE,1,2)

twoway (scatter SCHSIZE SC003Q01TA)











use "DNK 2018 Students.dta"

foreach x in READ SCIE MATH {
	gen `x' = (PV1`x'+PV2`x'+PV3`x'+PV4`x'+PV5`x'+PV6`x'+PV7`x'+PV8`x'+PV9`x'+PV10`x')/10
}

keep CNT CNTSCHID CNTSTUID ST001D01T ST003D02T ST003D03T ST004D01T ST005Q01TA ST007Q01TA ST022Q01TA WEALTH ESCS READ SCIE MATH





merge m:1 using "DNK Schools.dta"
keep CNT CNTSCHID PROAT5AB SC001Q01TA SC003Q01TA SC004Q01TA SC048Q02NA SC048Q03NA SCHLTYPE SCHSIZE STRATIO TOTAT