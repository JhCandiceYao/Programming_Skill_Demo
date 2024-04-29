clear all 

 * Load the 2018 GSS data
import delimited "/Users/lavender1/Desktop/SHIFTPROJECT/gss_data_f20.csv", delimiter(comma) clear 
* Selected variables of interest for regression controls: sex, race, satfin, parsol, kidssol

* code missing data by cross-referencing the codebook and actual data(tabulated first)
recode income(99 = .)
* reverse order for happiness and happiness of marriage so larger numerical value means higher degree of happiness
recode happy (8 = .) (1=3) (3=1)
recode hapmar (8 = .) (9 = .)(1=3) (3=1)
recode satfin (8 = .)
recode parsol (0 = .) (8 = .)
* for kids' living standards, 'NO CHILDREN -VOLUNTEERED-' is recoded as missing data so it will not bias regression analysis
recode kidssol (0 = .) (8 = .) (6 = .) 


* value-labeling for easier interpretation based on codebook
label define income_label ///
1 "Under $1,000" ///
2 "$1,000 to $2,999" ///
3 "$3,000 to $3,999" ///
4 "$4,000 to $4,999" ///
5 "$5,000 to $5,999" ///
6 "$6,000 to $6,999" ///
7 "$7,000 to $7,999" ///
8 "$8,000 to $9,999" ///
9 "$10,000 to $12,499" ///
10 "$12,500 to $14,999" ///
11 "$15,000 to $17,499" ///
12 "$17,500 to $19,999" ///
13 "$20,000 to $22,499" ///
14 "$22,500 to $24,999" ///
15 "$25,000 to $29,999" ///
16 "$30,000 to $34,999" ///
17 "$35,000 to $39,999" ///
18 "$40,000 to $49,999" ///
19 "$50,000 to $59,999" ///
20 "$60,000 to $74,999" ///
21 "$75,000 to $89,999" ///
22 "$90,000 to $109,999" ///
23 "$110,000 to $129,999" ///
24 "$130,000 to $149,999" ///
25 "$150,000 to $169,999" ///
26 "$170,000 or over"
label values income income_label

label define happy_label 1 "Not Too Happy" 2 "Pretty Happy" 3 "Very Happy"
label values happy happy_label
label values hapmar happy_label

label define sex_label 1 "Male" 2 "Female"
label values sex sex_label

label define race_label 1 "White" 2 "Black" 3 "Other"
label values race race_label

label define satfin_label 1 "Satisfied" 2 "More or less" 3 "Not at All Satisfied"
label values satfin satfin_label

label define sol_label 1 "Much better" 2 "Somewhat Better" 3 "About the Same" 4 "Somewaht Worse" 5 "Much Worse"
label values parsol sol_label
label values kidssol sol_label
* visualize distributions of income, happy, and hapmar
* save them after plotting the histograms
histogram income, discrete frequency addlabel normal title("Income Distribution") 
graph export "income_distribution.png", replace
histogram happy, discrete frequency addlabel normal title("General Happiness Distribution") 
graph export "happy_distribution.png", replace
histogram hapmar, discrete frequency addlabel normal title("Happiness of Marriage Distribution") 
graph export "happy_marriage_distribution.png", replace

* create cross-tabs to understand associations between happiness and income 
* display row and column percentages 
tabulate income happy, chi2 row col
* create cross-tabs to understand associations between happiness of marriage and income
* display row and column percentages 
tabulate income hapmar, chi2 row col

* fit OLS model and apply robust standard errors 
* include selected warranted variables
regress happy income sex race satfin parsol kidssol, vce(robust)
regress hapmar income sex race satfin parsol kidssol, vce(robust)



