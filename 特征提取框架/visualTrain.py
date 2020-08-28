import matplotlib.pyplot as plt

def visual(features,labels,epoch,cls):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    for k in range(4):
        plt.plot(features[labels == k], features[labels == k], '.', c=c[k])
    plt.legend(['0', '1', '2', '3'], loc='upper right')
    # plt.xlim(xmin=-100,xmax=100)
    # plt.ylim(ymin=-100,ymax=100)
    plt.title("epoch=%d,cls:%.4f " % (epoch, cls))
    plt.savefig('./visualtrain/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)