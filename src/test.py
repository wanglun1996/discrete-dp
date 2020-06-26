def binarySearch(x, a):
    l = 0
    u = len(x) - 1
    while u > l:
        idx = (l+u) / 2
        if x[idx] == a:
            return idx
        if x[idx] > a:
            u = idx - 1
            if l == u:
                return l
        if x[idx] < a:
            l = idx + 1
            if l == u:
                if x[l] <= a:
                    return l
                else:
                    return l-1

a = [1, 2, 3, 4, 5]
print(binarySearch(a, 1))
print(binarySearch(a, 1.5))
print(binarySearch(a, 2))
print(binarySearch(a, 2.5))
print(binarySearch(a, 4.9))
