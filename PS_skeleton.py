#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pandorable problem set 3 for PSY 1210 - Fall 2019

@author: katherineduncan

In this problem set, you'll practice your new pandas data management skills, 
continuing to work with the 2018 IAT data used in class

Note that this is a group assignment. Please work in groups of ~4. You can divvy
up the questions between you, or better yet, work together on the questions to 
overcome potential hurdles 
"""

#%% import packages 
import os
import numpy as np
import pandas as pd

#%%
# Question 1: reading and cleaning

# read in the included IAT_2018.csv file
IAT_data ='/Users/katherinebak/Desktop/Grad School Classes/PSY1210-Programming/Problem Set 3/pandorable/IAT_2018.csv'
IAT = pd.read_csv(IAT_data) 

# rename and reorder the variables to the following (original name->new name):
# session_id->id
# genderidentity->gender
# raceomb_002->race
# edu->edu
# politicalid_7->politic
# STATE -> state
# att_7->attitude 
# tblacks_0to10-> tblack
# twhites_0to10-> twhite
# labels->labels
# D_biep.White_Good_all->D_white_bias
# Mn_RT_all_3467->rt

IAT_renamed = IAT.rename(columns={'session_id':'id' , 'genderidentity':'gender','raceomb_002':'race','edu':'edu',
                        'politicalid_7':'politic','STATE':'state','att_7':'attitude','tblacks_0to10':'tblack',
                        'twhites_0to10':'twhite','labels':'labels','D_biep.White_Good_all':'D_white_bias','Mn_RT_all_3467':'rt'}) #renames each variable. first name listed is the original name, second name listed is the new name

IAT_renamed_reordered = IAT_renamed[['id','gender', 'race', 'edu', 'politic', 'state',
                                     'attitude', 'tblack', 'twhite', 'labels','D_white_bias', 'rt']] #reorders columns in this specific order 

# remove all participants that have at least one missing value
IAT_clean = IAT_renamed_reordered.dropna(axis =0, how='any',inplace=False) #taking the rows (axis=0) and droping any rows that have NA values (whether they say NAN or have no value)


# check out the replace method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
# use this to recode gender so that 1=men and 2=women (instead of '[1]' and '[2]')
IAT_clean = IAT_clean.replace('[1]','1')#will replace [1] with 1 
IAT_clean = IAT_clean.replace('[2]','2')#""and [2] with 2

# use this cleaned dataframe to answer the following questions

#%%
# Question 2: sorting and indexing

# use sorting and indexing to print out the following information:

# the ids of the 5 participants with the fastest reaction times
    #first sort rt to be fastest to slowest and then index the first 5 rows and then print the ids of these 5 people 

IAT_sorted_rt=IAT_clean.sort_values(by="rt", ascending = True) #sorting values so that rt is going from fastest(or smallest number) to slowest rt 
ids_fastest_rt=IAT_sorted_rt.iloc[0:5,0] #sorting values so that you are taking the first 5 rows from column 0 which is id
print(ids_fastest_rt) #will print ids of the 5 fastest participants 

# the ids of the 5 men with the strongest white-good bias
    #so get data with only men, then sort by highest white-good bias, then index first 5 rows, then print ids

IAT_men1=IAT_clean[IAT_clean.gender=='1'] #get data with only men,
IAT_men_wgb=IAT_men1.sort_values(by=["D_white_bias"],ascending = False) #sort by highest white-good bias, 
ids_m_highest_wgb=IAT_men_wgb.iloc[0:5,0]#then index first 5 rows,
print(ids_m_highest_wgb)# then print ids of these people 


# the ids of the 5 women in new york with the strongest white-good bias

IAT_women_ny=IAT_clean[(IAT_clean.gender=='2') & (IAT_clean.state=='NY')] #get data with only women and NY
IAT_women_ny_wgb=IAT_women_ny.sort_values(by=["D_white_bias"], ascending = False)#sort by highest white-good-bias
ids_w_highest_wgb=IAT_women_ny_wgb.iloc[0:5,0]#then index the first five rows
print(ids_w_highest_wgb) #prints ids of these people 


#%%
# Question 3: loops and pivots

# check out the unique method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html
# use it to get a list of states


states =pd.Series(pd.Categorical(IAT_clean.state)).unique() #makes a variable in a series with just the unique states in it




# write a loop that iterates over states to calculate the median white-good
# bias per state
# store the results in a dataframe with 2 columns: state & bias

df_states_bias=pd.DataFrame(columns=['state','bias']) #makes blank dataframe with only two columns for the loop to put its data into 


for state in states:
    median = IAT_clean[IAT_clean.state == state].D_white_bias.median() #get the median of each state white good-bias 
    df_states_bias = df_states_bias.append({'state': state, 
                                    'bias': median}, ignore_index=True) #put the value found above(after each loop) in the correct columns in the dataframe (the blank one from above))
df_states_bias=df_states_bias.sort_values(by=['state'])



# now use the pivot_table function to calculate the same statistics
state_bias=pd.pivot_table(IAT_clean,values=['D_white_bias'],
                          index=['state'],
                          aggfunc=np.median)
state_bias
# make another pivot_table that calculates median bias per state, separately 
# for each race (organized by columns)
state_race_bias=pd.pivot_table(IAT_clean, values=['D_white_bias'],
                               index=['state'],
                               columns=['race'],
                               aggfunc=np.median)

#%%
# Question 4: merging and more merging

# add a new variable that codes for whether or not a participant identifies as 
# black/African American

IAT_clean['black_AA']=1*(IAT_clean.race==5)#creates new variable called black_AA and codes it as either 1 or 0 with 1 meaning their race is black/AA and 0 meaning it is not

# use your new variable along with the crosstab function to calculate the 
# proportion of each state's population that is black 
# *hint check out the normalization options
prop_black =count=(pd.crosstab(IAT_clean.state, IAT_clean.black_AA==1, normalize='index'))*100#pmoves decimal over two places

prop_black=prop_black.loc[:,True] #taking only the column labeled True in prop_black
prop_black=prop_black.rename('prop_black') #renames this column as prop_black

# state_pop.xlsx contains census data from 2000 taken from http://www.censusscope.org/us/rank_race_blackafricanamerican.html
# the last column contains the proportion of residents who identify as 
# black/African American 
# read in this file and merge its contents with your prop_black table
census_file = '/Users/katherinebak/Desktop/Grad School Classes/PSY1210-Programming/Problem Set 3/pandorable/state_pop.xlsx'

census_file=pd.read_excel(census_file) 

merged=pd.merge(census_file, prop_black, left_on='State',right_on='state') #took right and left columns of each file to merge them as one (so there is only one state column in the new data)


# use the corr method to correlate the census proportions to the sample proportions

census_sample_corr = merged.corr().loc['per_black', 'prop_black'] #correlation of census prop black to sample prop black 

# now merge the census data with your state_race_bias pivot table

merged_race = pd.merge(census_file, state_race_bias, left_on='State', 
                       right_on='state') #taking the census data and merging it with the sata_race_bias pivot table. 

# use the corr method again to determine whether white_good biases is correlated 
# with the proportion of the population which is black across states
# calculate and print this correlation for white and black participants

census_race_corr = merged_race.corr().loc['per_black',('D_white_bias',5.0):('D_white_bias',6.0)] #correlating white good bias with prop of popuation which is black (5.0) and also white (6.0) across states

print(census_race_corr[('D_white_bias',5.0)])#black participants
print(census_race_corr[('D_white_bias',6.0)])#white participants

