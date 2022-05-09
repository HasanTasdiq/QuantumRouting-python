import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.transforms
# import latex
import matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

class ChartGenerator:
    # data檔名 Y軸名稱 X軸名稱 Y軸要除多少(10的多少次方) Y軸起始座標 Y軸終止座標 Y軸座標間的間隔
    def __init__(self, dataName, Ylabel, Xlabel):
        # Ydiv, Ystart, Yend, Yinterval
        Ydiv = 0
        div = int(10 ** Ydiv)
        print("start generator")
        color = [
            "#000000",
            "#006400",
            "#FF1493",
            "#7FFF00",   
            "#900321",
            "#000000",
            "#000000",
            "#000000",
            "#000000",
        ]
        # matplotlib.rcParams['text.usetex'] = True
        Xlabel_fontsize = 38
        Ylabel_fontsize = 38
        Xticks_fontsize = 38
        Yticks_fontsize = 38

        plt.rcParams["font.family"] = "Times New Roman"
            
        # matplotlib.rcParams['text.usetex'] = True
        # fig, ax = plt.subplots(figsize=(8, 6), dpi=600) 
        
        andy_theme = {
        # "axes.grid": True,
        # "grid.linestyle": "--",
        # "legend.framealpha": 1,
        # "legend.facecolor": "white",
        # "legend.shadow": True,
        # "legend.fontsize": 14,
        # "legend.title_fontsize": 16,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        # "text.usetex": True,
        # "figure.dpi": 100,
        }
        matplotlib.rcParams.update(andy_theme)
        fig, ax1 = plt.subplots(figsize=(9, 6), dpi=600)
        
        filename = './data/' + dataName
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        ##data start##
        x = []
        _y = []
        numOfData = 0



        for line in lines:
            line = line.replace('\n','')
            data = line.split(' ')
            numOfline = len(data)
            numOfData += 1
            for i in range(numOfline):
                if i == 0:
                    x.append(data[i])
                else:
                    _y.append(data[i])        
        
        numOfAlgo = len(_y) // numOfData

        y = [[] for _ in range(numOfAlgo)]
        for i in range(numOfData * numOfAlgo):
            y[i % numOfAlgo].append(_y[i])

        print(x)
        print(y)

        maxData = 0
        minData = math.inf

        for i in range(numOfAlgo):
            for j in range(numOfData):
                y[i][j] = float(y[i][j]) / div
                maxData = max(maxData, y[i][j])
                minData = min(minData, y[i][j])

        Ystart = int(max(0, minData - 5))
        Yend = int(maxData + 5)
        Yinterval = int((Yend - Ystart) // 5)

        marker = ['o', 's', 'v', 'x', 'd']
        for i in range(numOfAlgo):
            ax1.plot(x, y[i],color=color[i],lw=2.5,linestyle="-",marker=marker[i],markersize=20,markerfacecolor="none",markeredgewidth=2.5)
        # plt.show()
        plt.xticks(fontsize=Xticks_fontsize)
        plt.yticks(fontsize=Yticks_fontsize)
        
        AlgoName = ["MyAlgo", "Greedy_hop", "Greedy_geo", "QCAST", "REPS"]

        leg = plt.legend(
            AlgoName[0 : numOfAlgo],
            loc=10,
            bbox_to_anchor=(0.4, 1.1),
            prop={"size": "15", "family": "Times New Roman"},
            frameon="False",
            labelspacing=1,
            handletextpad=0.2,
            handlelength=1,
            columnspacing=0.5,
            ncol=4,
            facecolor="None",
        )

        leg.get_frame().set_linewidth(0.0)
        
        Ylabel += self.genMultiName(Ydiv)
        
        plt.yticks(np.arange(Ystart, Yend, step=Yinterval), fontsize=Yticks_fontsize)
        plt.ylabel(Ylabel, fontsize = Ylabel_fontsize)
        plt.xlabel(Xlabel, fontsize = Xlabel_fontsize)
        # plt.show()
        plt.tight_layout()
        pdfName = dataName[0:-4]
        plt.savefig('./pdf/{}.pdf'.format(pdfName)) 
        # Xlabel = Xlabel.replace(' (%)','')
        # Xlabel = Xlabel.replace('# ','')
        # Ylabel = Ylabel.replace('# ','')
        plt.close()

    def genMultiName(self, multiple):
        if multiple == 0:
            return str()
        else:
            return "($" + "\\times 10" + "^{" + str(multiple) + "}" + "$)"

if __name__ == "__main__":
    # data檔名 Y軸名稱 X軸名稱 Y軸要除多少(10的多少次方) Y軸起始座標 Y軸終止座標 Y軸座標間的間隔
    # ChartGenerator("numOfnodes_waitingTime.txt", "need #round", "#Request of a round", 0, 0, 25, 5)
    Xlabels = ["#RequestPerRound", "totalRequest", "#nodes", "r", "swapProbability", "alpha", "SocialNetworkDensity"]
    Ylabels = ["algorithmRuntime", "waitingTime", "unfinishedRequest", "idleTime", "usedQubits", "temporaryRatio"]
    
    for Xlabel in Xlabels:
        for Ylabel in Ylabels:
            dataFileName = Xlabel + '_' + Ylabel + '.txt'
            ChartGenerator(dataFileName, Ylabel, Xlabel)