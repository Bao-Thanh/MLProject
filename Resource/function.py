import library as lb


# Hàm trả về tần số xuất hiện  của các sample
# trong 1 feature non-numeric
# x là label , y là value
def tanso_featurenonnumeric(features):
    x = []
    y = []
    features = features
    freqs = lb.Counter(features)
    temp = lb.np.array(list(freqs.items()))
    for i in range(1,len(freqs.values())):
        x.append(temp[i][0])
        y.append(temp[i][1])
    return lb.np.array(x), lb.np.array(y)
