
from matplotlib import pyplot as plt
import xlrd
from matplotlib_venn import venn3
import pandas as pd


def save_res(acc, test_auc, sensitivity, specificity, k, cl):
    """
        save the results of the experiments into a csv file
    """
    acc.append(calc_avg(acc))
    test_auc.append(calc_avg(test_auc))
    sensitivity.append(calc_avg(sensitivity))
    specificity.append(calc_avg(specificity))
    header = ['fold' + str(n) for n in range(k)]
    header.append('average')
    print(acc)
    print(test_auc)
    print(header)
    print(sensitivity)
    print(specificity)
    dataframe = pd.DataFrame(
        {'': header, 'ACC': acc, 'sensitivity': sensitivity, 'specificity': specificity, 'AUC': test_auc})
    dataframe.to_csv('res_' + str(k) + 'fold_' + str(cl) + 'clusters.csv', index=False, sep=',')


def calc_avg(listx):
    """
        calculate the average value of a list
    """
    avg = 0
    for i in range(len(listx)):
        avg = avg + listx[i]
    avg = avg / (len(listx))
    return avg


def plot_venn():
    data = xlrd.open_workbook('ANN_venn.xls')
    table = data.sheet_by_name('80-40')
    set1 = set(table.row_values(1))
    set2 = set(table.row_values(2))
    set3 = set(table.row_values(3))
    # set1.remove('')
    # set3.remove('')
    print(set1)
    print(set2)
    print(set3)
    plt.figure(figsize=(15, 15))
    venn3([set1, set2, set3], ('ALL', 'IMRT', 'Protons'))
    plt.title("2-layer-ANN-40-cluster Venn diagram")
