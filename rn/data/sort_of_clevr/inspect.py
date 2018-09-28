import matplotlib.pyplot as plt

from rn.data.sort_of_clevr import generator


sort_of_clevr = generator(seed=80085)

for img, _, _, question, answer in sort_of_clevr:
    print("Question: ", question)
    print("Answer:", answer)
    print("===============================================")
    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress(0)
