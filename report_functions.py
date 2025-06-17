import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def set_font(font='Times New Roman'):
    plt.rcParams['font.family'] = font

def plot_f_temporal(path,methods,t_max, ARM_labels = [], figsize = (8,5)):
    plt.figure(1,figsize = figsize)

    if 'MC' in methods:
        MC = 1.-np.mean(np.load(path + 'MC.npy')[:,:,0], axis=1)
        plt.scatter(range(0,t_max+1,10), MC[:t_max+1:10], label = 'MC', marker = 'o', edgecolors='black' ,c = 'None', zorder=10)
    if 'PA' in methods:
        PA = 1.-np.mean(np.load(path + 'PA.npy')[:,:,0], axis=1)
        plt.plot(range(t_max+1),  PA[:t_max+1] , label = 'PA')
    if 'DMP' in methods:
        DMP = 1.-np.mean(np.load(path + 'DMP.npy')[:,:,0], axis=1)
        plt.plot(range(t_max+1),  DMP[:t_max+1], label = 'DMP',linestyle='--',dashes = (8,8))

    if 'ARM' in methods:
        colors = plt.cm.tab10.colors
        for label in ARM_labels:
            data = 1.-np.mean(np.load(path + 'ARM_' + label + '.npy')[:,:,0], axis=1)
            plt.plot(range(t_max+1), data[:t_max+1],color = colors[int(label[-1])-1], label = label)
        
    plt.ylabel('f',style='italic')
    plt.xlabel('Time $t$')
    
    if 'PA' in methods:
        y_max = np.ceil(100*np.max(PA))/100.
    elif 'DMP' in methods:
        y_max = np.ceil(100*np.max(DMP))/100.
    else:
        y_max = np.ceil(100*np.max(MC))/100.
    plt.ylim(0,y_max)
    plt.yticks(np.arange(0,y_max,0.1))
    plt.xlim(0,t_max)

def plot_error_temporal(path,methods,t_max, ARM_labels = [],figsize = (8,5)):
    plt.figure(1,figsize = figsize)
    MC = np.load(path + 'MC.npy')[:,:,0]
    if 'PA' in methods:
        PA = np.mean(np.abs(np.load(path + 'PA.npy')[:,:,0] - MC), axis=1)
        plt.plot(range(t_max+1),  PA[:t_max+1] , label = 'PA')
    if 'DMP' in methods:
        DMP = np.mean(np.abs(np.load(path + 'DMP.npy')[:,:,0] - MC), axis=1)
        plt.plot(range(t_max+1),  DMP[:t_max+1] , label = 'DMP',linestyle='--',dashes = (8,8))
    
    y_min = 1
    if 'ARM' in methods:
        colors = plt.cm.tab10.colors
        for label in ARM_labels:
            data = np.mean(np.abs(np.load(path + 'ARM_' + label + '.npy')[:,:,0] - MC), axis=1)
            y_min = min(y_min,np.max(data[:t_max+1]))
            plt.plot(range(t_max+1), data[:t_max+1], color = colors[int(label[-1])-1], label = label)
    
    plt.ylabel('$L_1$ Error')

    plt.xlabel('Time $t$')

    if 'PA' in methods:
        y_max = np.ceil(100*np.max(np.max(PA)))/100.

    if y_min < 0.01:
        y_min = -0.01
    else:
        y_min = 0
    plt.ylim(y_min,y_max)
    plt.xlim(0,t_max)
    plt.legend()

def plot_density(path, methods, t, ARM_labels = [], bins=10 , figsize = (8,5)):
    m_min = 0.
    m_max = 1.
    bins = np.linspace(m_min,m_max,bins+1)
    plt.figure(1,figsize = figsize)

    if 'MC' in methods:
        MC = 1.-np.load(path + 'MC.npy')[t,:,0]
        MC = norm.pdf(bins, np.mean(MC), np.std(MC))
        plt.plot(bins,  MC,'black', label = 'MC')
    if 'PA' in methods:
        PA = 1.-np.load(path + 'PA.npy')[t,:,0]
        PA = norm.pdf(bins, np.mean(PA), np.std(PA))
        plt.plot(bins,  PA , label = 'PA')
    if 'DMP' in methods:
        DMP = 1.-np.load(path + 'DMP.npy')[t,:,0]
        DMP = norm.pdf(bins, np.mean(DMP), np.std(DMP))
        plt.plot(bins,  DMP, label = 'DMP',linestyle='--',dashes = (8,8))

    if 'ARM' in methods:
        colors = plt.cm.tab10.colors
        for label in ARM_labels:
            data = 1.-np.load(path + 'ARM_' + label + '.npy')[t,:,0]
            data = norm.pdf(bins, np.mean(data), np.std(data))
            plt.plot(bins, data ,color=colors[int(label[-1])-1], label = label)

    plt.xlim(m_min,m_max)
    plt.ylim(0,)
    plt.xlabel('C')
    plt.ylabel('Density')

    plt.legend()

def scatter_error(path, t, n_point=0, label_ARM='', figsize=(8, 5)):
    plt.figure(1, figsize=figsize)
    
    MC = 1 - np.load(path + 'MC' + '.npy')[t, :, 0][1:] # removing the zero patient
    ARM = 1 - np.load(path + 'ARM_' + label_ARM + '.npy')[t, :, 0][1:] 
    PA = 1 - np.load(path + 'PA' + '.npy')[t, :, 0][1:] 
    
    if n_point != 0 & n_point<len(MC):
        random = np.random.choice(len(MC), n_point, replace=False)
        MC = MC[random]
        PA = PA[random]
        ARM = ARM[random]

    v_min = np.min(MC)
    v_max = max(np.max(MC), np.max(PA), np.max(ARM))

    colors = plt.cm.tab10.colors

    plt.scatter(PA, MC, label='PA', marker='o', edgecolors=colors[0], c='None', zorder=10,s=25)
    plt.scatter(ARM, MC, label=label_ARM, marker='+', color=colors[int(label_ARM[-1])-1], zorder=9,s=15)
    plt.plot([v_min, v_max], [v_min, v_max], color='green', linestyle='--')

    plt.xlim(v_min, v_max)
    plt.ylim(v_min, v_max)
    plt.xlabel('Method')
    plt.ylabel('MC')
    plt.legend()

def print_late_time_CI(path,g_name,methods,label_ARM,t=-1,precision=5):
    print(f'Network:{g_name}')
    if 'MC' in methods:
        MC = 1 - np.load(path + 'MC.npy')[t, :, 0]
        print('MC:', np.round(np.average(MC), precision),end = ', ')
    if 'DMP' in methods:
        DMP = 1 - np.load(path + 'DMP.npy')[t, :, 0]
        print('DMP:', np.round(np.average(DMP), precision),end = ', ')
    if 'PA' in methods:
        PA = 1 - np.load(path + 'PA.npy')[t, :, 0]
        print('PA:', np.round(np.average(PA), precision),end = ', ')
    if 'ARM' in methods:
        ARM = 1 - np.load(path + 'ARM_' + label_ARM + '.npy')[t, :, 0]
        print('ARM' + ':', np.round(np.average(ARM), precision))