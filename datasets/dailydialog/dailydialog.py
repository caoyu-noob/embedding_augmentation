with open('dialogues_topic.txt', 'r', encoding='utf-8') as f:
    topics = f.readlines()

with open('dialogues_text.txt', 'r', encoding='utf-8') as f:
    texts = f.readlines()
topic_dict = {}
for topic, line in zip(topics, texts):
    topic_dict[line] = topic

def process(data):
    cnt = 0
    new_train = []
    for i, line in enumerate(data):
        if not topic_dict.__contains__(line):
            cnt += 1
            continue
        topic = topic_dict[line]
        line = line.replace(' __eou__ ', '\t')
        line = line.replace('__eou__', '\t')
        if line[-2] == '\t':
            line = line[:-2] + '\n'
        new_train.append('<topic' + topic[:-1] + '>\t' + line)
    return new_train, cnt

with open('dialogues_train.txt', 'r', encoding='utf-8') as f:
    train = f.readlines()
new_train, cnt = process(train)
with open('dialogues_validation.txt', 'r', encoding='utf-8') as f:
    valid = f.readlines()
new_valid, cnt = process(valid)
with open('dialogues_test.txt', 'r', encoding='utf-8') as f:
    test = f.readlines()
new_test, cnt = process(test)
with open('train_data.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_train)
with open('valid_data.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_valid)
with open('test_data.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_test)
