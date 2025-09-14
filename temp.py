for i in range(100):
    for j in player.memory_many[i].samples:
        if abs(j[2] > 10):
            print(i)
            print(j)