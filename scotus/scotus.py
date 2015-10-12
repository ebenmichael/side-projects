# -*- coding: utf-8 -*-
#good site: http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_03_statistical_graphs.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

rcParams['patch.edgecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['xtick.major.width'] = 0

data = pd.read_excel("SCDB_2013_01_caseCentered_Citation.xlsx", 'Data')

maj_op = data[['chief','majVotes']]
chiefs = data['chief'].unique()

vinson = data[data['chief'] == 'Vinson']
warren = data[data['chief'] == 'Warren']
burger = data[data['chief'] == 'Burger']
rehnquist = data[data['chief'] == 'Rehnquist']
roberts = data[data['chief'] == 'Roberts']

justices = [vinson,warren,burger,rehnquist,roberts]


figure = plt.figure()
axes = figure.add_subplot(1, 1, 1, axisbg='.75')
vinson['majVotes'].hist(range = (4,9), bins = 5, normed = True, edgecolor = 'white', alpha = .2, label = 'Vinson')
warren['majVotes'].hist(range = (4,9), bins = 5, normed = True, edgecolor = 'white', alpha = .2, label = 'Warren')
burger['majVotes'].hist(range = (4,9), bins = 5, normed = True, edgecolor = 'white', alpha = .2, label = 'Burger')
rehnquist['majVotes'].hist(range = (4,9), bins = 5, normed = True, edgecolor = 'white', alpha = .2, label = 'Rehnquist')
roberts['majVotes'].hist(range = (4,9), bins = 5, normed = True, edgecolor = 'white', alpha = .2, label = 'Roberts')
axes.grid(False)
axes.yaxis.set_ticks_position('none')
axes.xaxis.set_ticks_position('none')
axes.set_xlabel('Justices voting in majority')
axes.set_title('Vinson\'s Court')
plt.legend(loc = 2)



"""ideology of decisions"""


issueCodes = ['Criminal Procedure','Civil Rights','Civil Rights',
'Due Process','Privacy','Attorneys','Unions','Economic Activity','Judicial Power',
'Federalism','Interstate Relations','Federal Taxation','Miscellaneous','Private Action']

issueCodesRed = ['Criminal Procedure','Civil Rights','Civil Rights',
'Due Process','Privacy','Attorneys','Unions','Economic Activity','Judicial Power',
'Federalism','Federal Taxation']

issues = data.groupby(['issueArea','decisionDirection']).size().unstack()
issues = issues.drop(issues.index[[10,12,13]]).drop(3,1).fillna(0)
#issues = (issues.T/issues.T.sum()).T # normalize
 
issuesJustice = []
for j in justices:
    justice = j.drop(j.index[j.issueArea == 14])
    justice = justice.drop(justice.index[justice.issueArea == 13])
    justice = justice.drop(justice.index[justice.issueArea == 11])
    issue = justice.groupby(['issueArea','decisionDirection']).size().unstack()
    issue = issue.fillna(0)
    #issue = (issue.T/issue.T.sum()).T #normalize
    issuesJustice.append(issue)


red,blue,green =  '#e41a1c', '#377eb8', '#4daf4a'
pp = PdfPages('issues.pdf')
plt.bar(np.linspace(0,10,11), issues[1],color = red, label = 'Conservative')
plt.bar(np.linspace(0,10,11), issues[2], bottom = issues[1], color = blue, label = 'Liberal')
plt.xticks(np.linspace(.5,10.5,11), issueCodesRed, rotation = 'vertical')
remove_border()
plt.legend(prop={'size':14})
plt.title('Ideology of Decisions in Supreme Court: Vinson - Roberts')
plt.ylabel('Number of Cases')
ax = plt.gca()
ax.xaxis.set_tick_params(width=0)
plt.tight_layout()
plt.axis('tight')
pp.savefig()
plt.close()





for i,chief in enumerate(chiefs):
    chiefissue = issuesJustice[i]
    plt.bar(np.linspace(0,10,11), chiefissue[1],color = red, label = 'Conservative')
    plt.bar(np.linspace(0,10,11), chiefissue[2], bottom = chiefissue[1], color = blue, label = 'Liberal')
    plt.xticks(np.linspace(.5,10.5,11), issueCodesRed, rotation = 'vertical')
    ax = plt.gca()
    ax.xaxis.set_tick_params(width=0) 
    remove_border()
    plt.ylabel('Number of Cases')
    plt.title('Ideology of Decisions in ' + chief + '\'s Court')
    plt.legend(prop={'size':14})
    plt.tight_layout()
    plt.axis('tight')
    pp.savefig()
    plt.close()
    
pp.close()  


"""Justice Centered"""

justiceData = pd.read_excel('SCDB_2013_01_justiceCentered_Citation.xlsx','Data')

