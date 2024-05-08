import random
from collections import defaultdict

#基于标注实体从TKG中检索特定问题的时间范围（将事实涉及时间戳进行排序，保留开始时间T1、结束时间T2）
def get_ents2reltime(kg_file):
    with open(kg_file) as f:
        #Python strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        #注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        # s.strip(rm)
        # 删除s字符串中开头、结尾处，位于rm删除序列的字符
        # s.lstrip(rm)
        # 删除s字符串中开头处，位于rm删除序列的字符
        # s.rstrip(rm)
        # 删除s字符串中结尾处，位于rm删除序列的字符
        # 注意：
        # （1）当rm为空时，默认删除空白符（包括'\n', '\r', '\t', ' ')
        #Python中的 split() 通过指定分隔符对字符串进行切片
        facts = f.read().strip().split('\n')
    facts = [f.strip().split('\t') for f in facts]
    #defaultdict里面的value类型会变为这个指定的数据类型
    # defaultdict还可以返回其他默认值，defaultdict一般都会接受一个工厂函数作为参数，不同的参数返回不同的默认值
    # 语法：dict = defaultdict(factory_function)   factory_function：工厂函数
    # 其中factory_function可以为set、list、str、int、dict等，当访问不存在的key
    # defaultdict(set)返回set()
    # defaultdict(list)返回[]
    # defaultdict(str)返回空字符串
    # defaultdict(int)返回0
    # defaultdict(dict) 返回dict

    #指定工厂函数后，value可以转变为相应的数据类型
    facts_di = defaultdict(set)
    #set是一种集合类型，它是一个无序的、可变的集合，包含不重复的元素。set是通过大括号 {} 来创建的，或者使用 set() 函数。
    for f in facts:
        facts_di[(f[0], f[2])].update([(f[0], f[1], f[2], f[3], f[4])])
    # facts_di[(f[0], f[2])].update( [(f[1], f[3], f[4])] )
    return facts_di

def get_ent2triplet(kg_file):
    with open(kg_file) as f:
        facts = f.read().strip().split('\n')
    facts = [f.strip().split('\t') for f in facts]
    facts_di = defaultdict(set)
    for f in facts:  # {key1: value1(set), key2: value2(set), key3: value3(set)}
        facts_di[(f[0])].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[(f[2])].update([(f[0], f[1], f[2], f[3], f[4])])
    # facts_di[(f[0], f[2])].update( [(f[1], f[3], f[4])] )
    return facts_di


def get_event2time(kg_file):
    with open(kg_file) as f:
        facts = f.read().strip().split('\n')
    facts = [f.strip().split('\t') for f in facts]
    facts_di = defaultdict(set)
    for f in facts:
        if f[1] == 'P793' and f[2] == 'Q1190554':
            facts_di[f[0]].update([(f[0], f[1], f[2], f[3], f[4])])
        # facts_di[f[0]].update( [( f[1], f[3], f[4])] )
    return facts_di


def get_ent_time2rel_ent(kg_file):
    with open(kg_file) as f:
        facts = f.read().strip().split('\n')
    facts = [f.strip().split('\t') for f in facts]
    facts_di = defaultdict(set)
    for f in facts:
        facts_di[(f[0], int(f[3]))].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[(f[0], int(f[4]))].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[(f[2], int(f[3]))].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[(f[2], int(f[4]))].update([(f[0], f[1], f[2], f[3], f[4])])
    return facts_di
#python中出现函数名相同但参数不同的函数时，后面的同名不同参的函数会覆盖掉前面的同名不同参函数，
def get_ent_time2rel_ent(kg_file):
    with open(kg_file) as f:
        facts = f.read().strip().split('\n')
    facts = [f.strip().split('\t') for f in facts]
    facts_di = defaultdict(lambda: defaultdict(set))
    for f in facts:
        facts_di[f[0]][int(f[3])].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[f[0]][int(f[4])].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[f[2]][int(f[3])].update([(f[0], f[1], f[2], f[3], f[4])])
        facts_di[f[2]][int(f[4])].update([(f[0], f[1], f[2], f[3], f[4])])
    return facts_di


def get_kg_facts_for_datapoint(e, e2tr, e2rt, et2re, event2time, thresh, time_delta=10):
    #keys()函数以列表返回一个字典所有的键。
    keys = e['annotation'].keys()
    if ('head' in keys) and ('tail' in keys) and ('tail2' in keys):
        head, tail, tail2 = e['annotation']['head'], e['annotation']['tail'], e['annotation']['tail2']
        return e2rt[(head, tail)].union(e2rt[(head, tail2)].union(e2rt[(tail, tail2)]))
    elif ('head' in keys) and ('tail' in keys):
        head, tail = e['annotation']['head'], e['annotation']['tail']
        return e2rt[(head, tail)]
    elif ('event_head' in keys) and ('tail' in keys):

        event_occ = event2time[e['annotation']['event_head']]
        if len(event_occ) > 0:
            event = next(iter(event_occ))
        tail_facts = [f for time, facts in et2re[e['annotation']['tail']].items() for f in facts]
        # """
        if len(event_occ) > 0:
            tail_facts = [f for f in tail_facts if
                          (int(f[3]) >= (int(event[3]) - time_delta)) and (int(f[4]) <= (int(event[4]) + time_delta))]
        # tail_facts = [f for f in tail_facts if (int(f[0]) >= (int(event[0]) - time_delta)) and (int(f[1]) <= (int(event[1]) + time_delta))]
        tail_facts = random.sample(tail_facts, thresh - 1) if len(tail_facts) > (thresh - 1) else tail_facts
        # """
        return set(list(event_occ) + tail_facts)
    elif 'time' in keys:
        ent = e['annotation']['head'] if 'head' in keys else e['annotation']['tail']
        return et2re[ent][int(e['annotation']['time'])]
    else:
        if ('head' in keys):
            ent = e['annotation']['head']
        else:
            ent = e['annotation']['tail']

        return e2tr[ent]


def append_time_to_question(question, facts):
    if facts:
        q = ', ' + str(facts[0]) + ', ' + str(facts[1])
        question['annotation']['time1'] = facts[0]
        question['annotation']['time2'] = facts[1]
        question['paraphrases'][0] = question['paraphrases'][0] + q
        question['question'] += q
        question['template'] = question['template'] + ', {time1}, {time2}'


def retrieve_time_for_question(d, facts, corrupt_p):
    whether_to_corrupt = [0, 1]
    corrupt_probs = [(1 - corrupt_p), corrupt_p]
    facts = list(facts)
    if len(facts) > 0:
        d['fact'] = []
        for f in facts:
            # probability of corruption during QA
            if random.choices(whether_to_corrupt, corrupt_probs, k=1)[0] == 0:
                d['fact'].append([f[3], f[4]])
    else:
        d['fact'] = []
    return


def add_facts_to_data(data, corrupt_p, fuse, e2tr, e2rt, et2re, event2time, thresh=5):
    for d in data:
        facts = get_kg_facts_for_datapoint(d, e2tr, e2rt, et2re, event2time, thresh)
        facts = sorted(facts, key=lambda x: x[3])

        # remove `no_time' if corrupted
        facts = [x for x in facts if x != 9620]

        # TempoQR-att appends to the question
        if fuse == 'att':
            append_time_to_question(d, facts)
        else:
            retrieve_time_for_question(d, facts, corrupt_p)
    return data

def retrieve_times(kg_file, dataset_name, data, corrupt_p, fuse):
    # kg_file could involve a corrupt TKG
    kg_file = f'./data/{dataset_name}/kg/' + kg_file

    # collecting possible combinations of annotated entities/timestamps
    e2tr = get_ent2triplet(kg_file)
    e2rt = get_ents2reltime(kg_file)
    et2re = get_ent_time2rel_ent(kg_file)
    event2time = get_event2time(kg_file)

    # collect all the question-specific timestmaps
    data = add_facts_to_data(data, corrupt_p, fuse, e2tr, e2rt, et2re, event2time)

    return data
