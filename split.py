import splitfolders

# https://www.kaggle.com/datasets/kapillondhe/american-sign-language
# splitfolders.ratio("ASL_Dataset/Train", output="ASL", ratio=(.8, .1, .1), seed=1)

# https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet
splitfolders.ratio("Synthetic_ASL_Alphabet/Train_Alphabet", output="Synthetic_ASL", ratio=(.9, .1), seed=1)