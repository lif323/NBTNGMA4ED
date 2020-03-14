#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    a = [1, 2, 3]
    s = set()
    s.update(a)
    print(s)
    s.update([6, 7, 9])
    print(s)
