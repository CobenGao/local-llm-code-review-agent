#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/data_processor.py
============================
数据处理工具模块(AI 评审触发测试用)

这是一份故意写了 4 个经典 Python 反模式的示例文件,
用于验证 AI Code Review Agent 是否能够正确识别:
  ① 可变默认参数
  ② 魔鬼数字(Magic Number)
  ③ 裸异常捕获(bare except)
  ④ 命名规范不符合 PEP 8
"""

import time



def merge_records(new_items, result=[]):
    for item in new_items:
        result.append(item)
    return result



def wait_for_next_cycle():
    time.sleep(86400)



def safe_divide(a, b):
    try:
        return a / b
    except:
        pass



def ProcessData(D):
    return [x * 2 for x in D]
