from IPython.core.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
# %matplotlib inline
sns.set(style='ticks',context='poster')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# (1) A general guideline for GNN design
# Trade-off analysis for each design dimension in the design space
# Figure 3 in "Design Space for Graph Neural Networks"
from scipy.stats import rankdata, ttest_ind, f_oneway
from matplotlib.ticker import MaxNLocator

results_file_path = '../run/results/design_v1_grid_round1/agg/val.csv'
df = pd.read_csv(results_file_path)
df['epoch'] += 1
df.replace('skipconcat','skipcat',inplace=True)
df.replace('add','sum',inplace=True)


name_mapping = {'act': 'Activation', 'bn':'Batch Normalization', 'drop':'Dropout',
                'agg':'Aggregation', 'l_mp':'Message passing layers', 'l_pre':'Pre-process layers',
                'l_post': 'Post-process layers', 'stage': 'Layer connectivity',
                'lr': 'Learning rate', 'batch':'Batch size', 'optim': 'Optimizer',
                'epoch': 'Training epochs', 'ha': 'Has Activation', 'ln': 'Has L2Norm',
                'di': 'Inner Layer Dim', 'att': 'Has Attention', 'at': 'Attentiohn Type',
                'ah': 'Attention Heads Num'
               }

column = ['sample', 'format', 'dataset', 'task', 'trans', 'feature', 'label',
       'batch', 'l_pre', 'l_mp', 'l_post', 'stage', 'bn', 'act', 'drop', 'agg',
       'optim', 'lr', 'epoch']

def get_acc(df, name, ax, metric='acc', has_y=True):  # 'bn', axes[0, 0], 'rank_bar'
    df_selected = df[df['sample']==name].copy()  # 'bn'
    df_selected[column] = df_selected[column].fillna('Nan')

    column_temp = copy.deepcopy(column)
    column_temp.remove(name)  # 删除'bn'
    # 建立数据透视表 
    df_pivot = pd.pivot_table(df_selected, values='accuracy', index=column_temp, columns=[name], aggfunc=np.mean)
    accs_np = df_pivot.fillna(df_pivot.min()).values.round(2)
    options = df_pivot.columns.values
        
    ranks_raw = {'Model ID':[], 'Accuracy':[], 'Acc. Ranking':[], name_mapping[name]:[]}
    
    rank_np = np.zeros((accs_np.shape[0], accs_np.shape[1]))
    for i,row in enumerate(accs_np):
        # (1) rank is asceneding, so we neg the row; (2) rank start with 1 so we minus 1
        rank_base = -row
        med = np.median(rank_base)
        bias = 0.021
        for j in range(len(rank_base)):
            if abs(rank_base[j]-med) <= bias:
                rank_base[j] = med
        rank = rankdata(rank_base, method='min')
        for j in range(len(rank)):
            ranks_raw['Model ID'].append(i)
            ranks_raw['Accuracy'].append(accs_np[i,j])
            ranks_raw['Acc. Ranking'].append(rank[j])
            ranks_raw[name_mapping[name]].append(options[j])
            rank_np[i,j]=rank[j]

#     if metric == 'rank_bar':
#         p_value = f_oneway(*[rank_np[:,i] for i in range(rank_np.shape[1])])[1]
#         print(name, p_value, p_value<0.05, p_value<0.05/12)

    ranks_raw = pd.DataFrame(data=ranks_raw)     
    with sns.color_palette("muted"):
        if metric=='acc':
            splot = sns.violinplot(x=name_mapping[name], y="Accuracy",inner="box", data=ranks_raw, cut=0, ax=ax)
        elif metric=='rank_bar':
            splot = sns.barplot(x=name_mapping[name], y="Acc. Ranking",data=ranks_raw, ax=ax)
            ax.set_ylim(bottom=1)
            ax.set_yticks([1,2])
            ax.set_xlabel('',fontsize=48)
            if not has_y:
                ax.set_ylabel('',fontsize=48)
            else:
                ax.set_ylabel('Average',fontsize=48)
        elif metric=='rank_violin':
            sns.violinplot(x=name_mapping[name], y="Acc. Ranking",inner="box", data=ranks_raw, cut=0, ax=ax)
            ax.set_ylim(bottom=1)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if not has_y:
                ax.set_ylabel('',fontsize=48)
            else:
                ax.set_ylabel('Distribution',fontsize=48)
        ax.xaxis.label.set_size(48)
        ax.yaxis.label.set_size(48)
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(40)
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(40)
        
for names in [['bn','drop','act','agg','l_mp','stage'], ['l_pre','l_post','batch','lr','optim','epoch']]:
    col = 6
    row = 2
    f, axes = plt.subplots(nrows=row, ncols=col, figsize=(48, 10))
    
    for j,metric in enumerate(['rank_bar','rank_violin']):
        for i,name in enumerate(names):
            get_acc(df, name, axes[j, i], metric, has_y=i==0)
    f.text(-0.001, 0.5, 'Accuracy Ranking', ha='center', va='center', rotation='vertical', fontsize=48)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
#     f.savefig('figs/{}.png'.format(metric), dpi=150, bbox_inches='tight')
    
    plt.show()