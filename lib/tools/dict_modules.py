#!-*-coding:utf-8-*-

__all__ = ["split_dict"]

def split_dict( dict , value_every ):
    """
    辞書をグラフでプロットする際に、valueでdictをsplitする
    --------------------------------
    input :
        - dict
        - value_every : スプリットする歩幅
    return : list
    --------------------------------
    """
    arr = {}
    for i in range( len( value_every ) - 1 ):
        tmp_dict = {}
        for key , value in dict.items():
            if value > value_every[i] and value <= value_every[i+1]:
                tmp_dict[key] = value

        arr[ str( value_every[i] ) + "-" + str( value_every[i+1] ) ] = tmp_dict

    return arr
