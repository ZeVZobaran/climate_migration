cd "C:\Dropbox\Brazil electricity\data and do files\"

use Brazil_Electricity_data.dta
log using regs_final.log, replace
tsset AMC60ID decade

*see below for table 1

***Table 2--variable summaries

sum electricity_infrastructure f.pct_elect matlab_instrument f.totalrescap_house f.HDI_popave f.HDIlongevity_popave f.HDIrenda_popav f.HDIeducation_popave f.econ_active_pct f.employed_pct f.urbanemployed_pct f.ruralemployed_pct f.yearsschool_popave f.illiteracy f.renda_pc f.hc_pc f.migration f.life_expect_popave f.popdens f.urbanpctpop f.poverty_popave f.Theil_popave f.infant_mort_popave f.l4edu_popave f.runningwater_pcthouses f.san_pcthouses f.carspcthouses landslope if sample==1

***Table 3
reg electricity_infrastructure  matlab_instrument  yeardum* [aweight=area] if sample==1, cl(AMC60)
est store reg1
test matlab_instrument==0
areg electricity_infrastructure  matlab_instrument  yeardum*  [aweight=area] if sample==1, absorb(AMC60ID) cl(AMC60)
est store reg2
test matlab_instrument==0
areg electricity_infrastructure matlab_instrument yeardum* jungleyr* [aweight=area] if e(sample)==1, absorb(AMC60ID) cl(AMC60)
est store reg3
test matlab_instrument==0
estout reg1  reg2 reg3  using table3.txt, keep(matlab_instrument)  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace

*see below for table 4

*Table 5
areg matlab_instrument l.totalrescap_house    yeardum* [aweight=area], absorb(AMC60ID) cl(AMC60ID)
est store reg1

areg matlab_instrument l.HDI   yeardum* [aweight=area], absorb(AMC60ID) cl(AMC60ID)
est store reg2

estout  reg1  reg2 using table5.txt, keep(L.HDI L.totalrescap_house )  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace

***Tables 6 and 7
foreach var in totalrescap_house HDI_popave {

reg f.`var' electricity_infrastructure yeardum*  [aweight=area],  cl(AMC60ID)
est store `var'1

areg f.`var' electricity_infrastructure yeardum* [aweight=area], absorb(AMC60ID) cl(AMC60ID)
est store `var'2

areg f.`var' electricity_infrastructure yeardum* jungleyr* [aweight=area], absorb(AMC60ID) cl(AMC60ID)
est store `var'3

ivreg2 f.`var' (electricity_infrastructure=matlab_instrument) yeardum* [aweight=area] , cl(AMC60ID)
est store `var'4

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument)  yeardum* [aweight=area] , fe cl(AMC60ID)
est store `var'5

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument) jungleyr* yeardum* [aweight=area] , fe cl(AMC60ID)
est store `var'6

estout `var'1 `var'2 `var'3 `var'4 `var'5 `var'6  using `var'_tables67.txt, keep(electricity_infrastructure)  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace
}
estimates clear


*table 8 first 2 columns
***note--cars data only available for 1991 and 2000, so cross sect reg necessary
reg carspcthouses landslopetrend watertrend yeardum*
est store cars1
xtreg carspcthouses landslopetrend watertrend yeardum*, fe
est store cars2
estout cars1 cars2 using table8c1c2.txt, keep(landslopetrend watertrend)  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace



*table8
foreach var in  totalrescap_house HDI  {

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument) runningwater_pcthouses san_pcthouses watertrend landslopetrend jungleyr*   yeardum* [aweight=area] if sample==1, fe cl(AMC60ID)
est store `var'

}
estout totalrescap_house  HDI using table8_c3c4.txt, keep(electricity_infrastructure  runningwater_pcthouses san_pcthouses carspcthouses)  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace


*table9


xtivreg2 HDI (pct_elect=matlab_instrument) yeardum* [aweight=area] , fe cl(AMC60ID)
est store pct_elect1

xtivreg2 f.HDI (electricity_infrastructure=US_matlab_instrument) yeardum* [aweight=area], fe cl(AMC60ID)
est store USinst1

xtivreg2 totalrescap_house (pct_elect=matlab_instrument) yeardum* [aweight=area] , fe cl(AMC60ID)
est store pct_elect2

xtivreg2 f.totalrescap_house (electricity_infrastructure=US_matlab_instrument) yeardum* [aweight=area], fe cl(AMC60ID)
est store USinst2

estout pct_elect1 USinst1 pct_elect2 USinst2 using table9.txt, keep(pct_elect electricity_infrastructure )  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace

estimates clear


****Table 10
ren electricity_infrastructure elect

estimates clear
eststo:  xtivreg2 f.HDI (elect=matlab_instrument) water_budget yeardum* [aweight=area] , fe cl(AMC60ID) first  savefprefix(iv1)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv1elect  using HDI.rtf, replace label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) wslope_budget yeardum* [aweight=area] , fe cl(AMC60ID)  first savefprefix(iv3)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv3elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

eststo:  xtivreg2 f.HDI (elect=matlab_instrument) j_budget yeardum* [aweight=area] , fe cl(AMC60ID) first  savefprefix(iv2)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv2elect using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) water_budget j_budget  yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv4)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv4elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear


xtivreg2 f.HDI (elect=matlab_instrument) j_budget wslope_budget yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv5)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv5elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)

estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) wslope_budget water_budget yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv6)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv6elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear


xtivreg2 f.HDI (elect=matlab_instrument)  j_budget  wslope_budget water_budget yeardum* [aweight=area] , fe first cl(AMC60ID) savefprefix(iv7)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv7elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) wateryr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv8)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv8elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01)
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)

estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) jungleyr*  yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv9)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv9elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)
estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) wslopeyear* yeardum*  [aweight=area] , fe cl(AMC60ID) first savefprefix(iv10)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv10elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1)  title(2nd stage)
estimates clear


xtivreg2 f.HDI (elect=matlab_instrument) wateryr* jungleyr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv11)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv11elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1)  title(2nd stage)

estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) jungleyr*  wslopeyear*  yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv12)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv12elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)


estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) wslopeyear*  wateryr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv13)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv13elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)


estimates clear

xtivreg2 f.HDI (elect=matlab_instrument) wslopeyear* jungleyr* wateryr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv13)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv13elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)


estimates clear
xtivreg2 f.HDI (elect=matlab_instrument)  hydrohat_yeardum*  hydrohat2_yeardum* hydrohat3_yeardum* hydrohat4_yeardum* yeardum* [aweight=area] , fe first cl(AMC60ID)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv13elect  using HDI.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using HDI.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)

**
eststo:  xtivreg2 f.totalrescap_house (elect=matlab_instrument) water_budget yeardum* [aweight=area] , fe cl(AMC60ID) first  savefprefix(iv1)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv1elect  using totalrescap_house.rtf, replace label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) wslope_budget yeardum* [aweight=area] , fe cl(AMC60ID)  first savefprefix(iv3)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv3elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

eststo:  xtivreg2 f.totalrescap_house (elect=matlab_instrument) j_budget yeardum* [aweight=area] , fe cl(AMC60ID) first  savefprefix(iv2)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv2elect using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) water_budget j_budget  yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv4)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv4elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear


xtivreg2 f.totalrescap_house (elect=matlab_instrument) j_budget wslope_budget yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv5)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv5elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)

estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) wslope_budget water_budget yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv6)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv6elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear


xtivreg2 f.totalrescap_house (elect=matlab_instrument)  j_budget  wslope_budget water_budget yeardum* [aweight=area] , fe first cl(AMC60ID) savefprefix(iv7)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv7elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) drop(yeardum*)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) drop(yeardum*) title(2nd stage)
estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) wateryr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv8)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv8elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01)
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)

estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) jungleyr*  yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv9)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv9elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)
estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) wslopeyear* yeardum*  [aweight=area] , fe cl(AMC60ID) first savefprefix(iv10)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv10elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1)  title(2nd stage)
estimates clear


xtivreg2 f.totalrescap_house (elect=matlab_instrument) wateryr* jungleyr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv11)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv11elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1)  title(2nd stage)

estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) jungleyr*  wslopeyear*  yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv12)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv12elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)

estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) wslopeyear* wateryr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv13)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv13elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)


estimates clear

xtivreg2 f.totalrescap_house (elect=matlab_instrument) wslopeyear* jungleyr* wateryr* yeardum* [aweight=area] , fe cl(AMC60ID) first savefprefix(iv13)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv13elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)

estimates clear
xtivreg2 f.totalrescap_house (elect=matlab_instrument)  hydrohat_yeardum* hydrohat2_yeardum* hydrohat3_yeardum* hydrohat4_yeardum* yeardum* [aweight=area] , fe first cl(AMC60ID)
estadd scalar F1 =el(e(first), 3, 1)
esttab iv13elect  using totalrescap_house.rtf, append label notes depvars addnotes("...") title(First-stage) nogap nomtitles cells(b(star fmt(3)) se (par fmt(3))) stats(N r2 r2_p, fmt(%9.0g %9.2f %9.2f %9.2f) labels (Observations "R-sq")) star(* 0.10 ** 0.05 *** 0.01) 
esttab using totalrescap_house.rtf, append label nogap cells(b(star fmt(3)) se(par fmt(3))) star(* 0.10 ** 0.05 *** 0.01) stats(N r2 F1) title(2nd stage)
stop

ren elect electricity_infrastructure
**Table 11
foreach var in HDIlongevity HDIincome HDIeducation infant_mort grossincome_pc poverty {

areg f.`var' electricity_infrastructure yeardum* jungleyr* [aweight=area], absorb(AMC60ID) cl(AMC60ID)
est store `var'1

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument) jungleyr*  yeardum* [aweight=area] , fe cl(AMC60ID)
est store `var'2
}
estout HDIlongevity1 HDIlongevity2 HDIincome1 HDIincome2 HDIeducation1 HDIeducation2 infant_mort1 infant_mort2 grossincome_pc1 grossincome_pc2 poverty1 poverty2 using table11.txt, keep(electricity_infrastructure)  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace

*table12
foreach var in  econ_active_pct employed_pct urbanemployed_pct ruralemployed_pct {

areg f.`var' electricity_infrastructure yeardum* jungleyr* [aweight=area] if sample==1, absorb(AMC60ID) cl(AMC60ID)
est store `var'1

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument) jungleyr* yeardum* [aweight=area] if sample==1, fe cl(AMC60ID)
est store `var'2
}
estout econ_active_pct1  econ_active_pct2 employed_pct1 employed_pct2 urbanemployed_pct1 urbanemployed_pct2 ruralemployed_pct1 ruralemployed_pct2  using employment.txt, keep(electricity_infrastructure )  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace

*table 13 
foreach var in   illiteracy l4edu yearsschool hc_pc  {

areg f.`var' electricity_infrastructure yeardum* jungleyr* [aweight=area] if sample==1, absorb(AMC60ID) cl(AMC60ID)
est store `var'1

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument) jungleyr* yeardum* [aweight=area] if sample==1 , fe cl(AMC60ID)
est store `var'2
}
estout illiteracy1 illiteracy2 l4edu1 l4edu2 yearsschool1 yearsschool2 hc_pc1 hc_pc2 using humancapital.txt , keep(elect_ )  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace
*table 14
foreach var in  migration life_expect popdens urbanpctpop {

areg f.`var' electricity_infrastructure yeardum* jungleyr* [aweight=area] if sample==1, absorb(AMC60ID) cl(AMC60ID)
est store `var'1

xtivreg2 f.`var' (electricity_infrastructure=matlab_instrument) jungleyr* yeardum* [aweight=area] if sample==1, fe cl(AMC60ID)
est store `var'2
}
estout migration1  migration2 life_expect life_expect popdens1 popdens2 urbanpctpop1 urbanpctpop2 using populationchange.txt, keep(elect_ )  cells(b(star fmt(3))  se(par fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N) replace


******Tables 1 and 4*****
use topographic_data.dta
probit hydro Amazon waterdum lnfa_max ave_slope max_slope if year==1960
outreg2 hydro Amazon waterdum lnfa_max ave_slope max_slope using table1.txt, bdec(3) nocon

*Table 4
bys year region:  spearman rank_generationr rank_popdensr
bys year region:  spearman rank_generationr rank_pibpcr


