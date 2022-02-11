import matplotlib.pyplot as plt

def read_csv(path):
    with open(path + '.csv', 'r') as f:
        header = f.readline().rstrip('\n')
        fields = header.split(',')
        results = {}
        for field in fields:
            results[field] = []
        for l in f:
            vals = dict(zip(fields, map(float, l.rstrip('\n').split(','))))
            for field in fields:
                results[field].append(vals[field])
    return results

x = 'epoch'
noises = {
    0: '0_0',
    2: '2_5',
    4: '4_5',
    6: '6_5'}
h2f = {
    'rn50': 'ResNet50',
    'vits': 'VIT',
    'swint': 'SwinT',
    'mixer': 'MLP-Mixer',
    'gmlp': 'gMLP',
}
results = {}
for shortform, filename in noises.items():
    results[shortform] = read_csv(filename)

for n in noises.keys():
    plt.figure()
    miny, maxy = 100, 0
    for arch, label in h2f.items():
        plt.plot(results[n][x], results[n][arch], label=label)
        if results[n][arch] < miny:
            
    plt.ylim([60 - n * 2.5, 85 - n * 2.5])
    plt.ylabel('Top 1 test accuracy (%)')
    plt.xlabel('Epoch')
    plt.title(f'Test accuracy vs epoch at NL={n}')
    plt.legend()
plt.show()
