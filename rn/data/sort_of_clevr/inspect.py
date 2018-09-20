import matplotlib.pyplot as plt

from rn.data.sort_of_clevr import generator


sort_of_clevr = generator()


while True:
    img, question, answer, q_enc, a_enc = next(sort_of_clevr)

    # TODO: Fix weird behaviour when exiting.

    print("Question: ", question)
    print("Answer:", answer)
    print("===============================================")
    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
