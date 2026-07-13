with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

import re

result = [
    item
    for item in re.split(r'([,.:!@#%&*()-]|--|\s)', raw_text)
    if item.strip()
]

#### Creating token IDs

all_words = sorted(set(result))
print("Vocab size: ",len(all_words))

vocab = {token:index for index,token in enumerate(all_words)}

### 
# class SimpleTokenizer():
#     def __init__(self,vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}

#     def encode(self,text):
#         text_pre = [
#             item
#             for item in re.split(r'([,.:!@#%&*()-]|--|\s)', raw_text)
#             if item.strip()
#         ]

#         ids = [self.str_to_int[i] for i in text_pre]
#         return ids
    
#     def decode(self,ids):
#         text = " ".join([self.int_to_str[i] for i in ids])

#         final_text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
#         return final_text
    
texter = """
"The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." The word, on Mrs. Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. And it was not only the Mrs. Thwings who mourned. Had not the exquisite Hermia Croft, at the last Grafton Gallery show, stopped me before Gisburn's "Moon-dancers" to say, with tears in her eyes: "We shall not look upon its like again"?
"""

# tokenizer = SimpleTokenizer(vocab)
# ids_1 = tokenizer.encode(texter)
# print(ids_1)
# print("*************")
# print(tokenizer.decode(ids_1))

### Tokenization
import tiktoken  ### Byte Pair tokenizer used by GPT-2,3

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode(texter))
print("************")
print(tokenizer.decode(tokenizer.encode(texter)))
