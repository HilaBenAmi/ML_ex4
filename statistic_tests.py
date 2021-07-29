import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import scikit_posthocs as sp

# import experiments results
xls = pd.ExcelFile("C:\\Users\\hilon\\OneDrive - post.bgu.ac.il\\תואר שני\\סמ 1 - אביב 2021\\למידה חישובית\\עבודה 4 סיום\\results sheet.xlsx")
mean_teacher_df = pd.read_excel(xls, 'mean_teacher')[:60]
supervised_df = pd.read_excel(xls, 'supervised')

# Wilcoxon signed-rank test
for col in mean_teacher_df.columns:
	if 'Name' not in col and 'Time' not in col and 'Values' not in col and 'Cross' not in col:
		print(f"Metric name: {col}")
		mt_metric = mean_teacher_df[col]
		sup_metric = supervised_df[col]

		# combine two groups into one array
		data = np.array([mt_metric, sup_metric])
		# perform Nemenyi post-hoc test
		res = sp.posthoc_nemenyi_friedman(data.T)
		print(res)

		# compare samples
		stat, p = wilcoxon(mt_metric, sup_metric)
		print('Statistics=%.4f, p=%.4f' % (stat, p))
		# interpret
		alpha = 0.05
		if p > alpha:
			print('The algorithms yield same results, the same distribution (fail to reject H0)')
		else:
			print('Different algorithms (reject H0)')


### calculate the hyper parameters avg
# mean_teacher_df['hyper_params'] = mean_teacher_df['Hyper-Parameters Values'].apply(lambda x: eval(x))
# mt_hyper_params_df = pd.DataFrame(list(mean_teacher_df['hyper_params']))
# mt_df = mean_teacher_df.join(mt_hyper_params_df)
# hyper_params_avg_per_dataset = mt_df.groupby('Dataset Name').aggregate({
# 										 'cls_balance': lambda x: np.mean(x),
# 										 'confidence_thresh': lambda x: np.mean(x),
# 										 'learning_rate': lambda x: np.mean(x),
# 										 'teacher_alpha': lambda x: np.mean(x),
# 										 'unsup_weight': lambda x: np.mean(x)}).add_prefix('mt_avg_')
# for col in list(hyper_params_avg_per_dataset.columns):
# 	hyper_params_avg_per_dataset.loc['avg', col] = hyper_params_avg_per_dataset[col].mean()
# 	hyper_params_avg_per_dataset.loc['std', col] = hyper_params_avg_per_dataset[col].std()
#
# supervised_df['hyper_params'] = supervised_df['Hyper-Parameters Values'].apply(lambda x: eval(x))
# sup_hyper_params_df = pd.DataFrame(list(supervised_df['hyper_params']))
# sup_df = supervised_df.join(sup_hyper_params_df)
# sup_hyper_params_avg_per_dataset = sup_df.groupby('Dataset Name').aggregate({'learning_rate': lambda x: np.mean(x)}).add_prefix('sup_')
# for col in list(sup_hyper_params_avg_per_dataset.columns):
# 	sup_hyper_params_avg_per_dataset.loc['avg', col] = sup_hyper_params_avg_per_dataset[col].mean()
# 	sup_hyper_params_avg_per_dataset.loc['std', col] = sup_hyper_params_avg_per_dataset[col].std()
#
# hyper_params_avg_per_dataset = hyper_params_avg_per_dataset.join(sup_hyper_params_avg_per_dataset)
# hyper_params_avg_per_dataset.to_csv('./all_hyper_params_avg_std.csv')