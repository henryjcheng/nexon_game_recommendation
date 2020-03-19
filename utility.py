'''
This program contains functions that are ran in Jupyter Notebook.
The purpose is to clean up Notebook so they are more readable.
'''
import numpy as np
import pandas as pd

def eda_general(df):
    '''
    This function contains codes for EDA phase 1
    '''
    # number of records
    print(f'Number of records:          {df.shape[0]}')

    # number of players
    n_player = df['id'].nunique()
    print(f'Number of unique User ID:   {n_player}')

    # number of games
    n_game = df['game'].nunique()
    print(f'Number of games:            {n_game}')

    # number of actions
    print('\n\tAction Count: ')
    print(df['action'].value_counts())
    
    # number of players who made purchase
    n_purch_player = df[df['action'] == 'play']['id'].nunique()
    print(f'\nNumber of Purchasing Player: {n_purch_player}')
    print(f'% of Purchasing Player: {round(n_purch_player/n_player, 2) * 100}%')

    # summary statistics on play time
    print('\n\tSummary Statistics on Play Time:')
    print(df[df['action'] == 'play']['value'].describe())

    # summary stats on number of games played
    df_temp = df[df['action'] == 'play'][['id', 'game']].drop_duplicates().reset_index(drop=True)
    df_temp['game_freq'] = df_temp.groupby('id')['id'].transform('count')
    print('\n\tSummary Statistics on Number of Games Played:')
    print(df_temp['game_freq'].describe())

    # summary stats on number of games purchased
    df_temp = df[df['action'] == 'purchase'][['id', 'game', 'action']].drop_duplicates().reset_index(drop=True)
    df_temp['purch_freq'] = df_temp.groupby('id')['id'].transform('count')
    print('\n\tSummary Statistics on Number of Games Purchase:')
    print(df_temp['purch_freq'].describe())

    # most purchased games
    df_temp = df[df['action'] == 'purchase'][['id', 'game']].drop_duplicates().reset_index(drop=True)
    print('\n\tTop 10 Games Purchased')
    print(df_temp['game'].value_counts()[:10])
    
    pass

def flag_nexon_player(df):
    '''
    This function flags Nexon player. The program creates 4 flags, each set to 1 if 
    player has 'player' action for one of the Nexon game. The last flag is set to 1 if 
    any Nexon game flag is set to 1.
    
    We also create a column for player type, with value:
    MapleStory, Mabinogi, Vindictus, Nexon, non_Nexon
    where 'MapleStory' indicates player has only played MapleStory game, ...etc.
    'Nexon' indicates player has played more than one Nexon game, and 'non_Nexon' means
    player has not played any Nexon game before
    '''
    # set flag for each Nexon game
    df_MS = df[(df['game'] == 'MapleStory') & (df['action'] == 'play')]['id'].to_frame().drop_duplicates()
    df_MS['flg_MS'] = 1
    df_MA = df[(df['game'] == 'Mabinogi') & (df['action'] == 'play')]['id'].to_frame().drop_duplicates()
    df_MA['flg_MA'] = 1
    df_VN = df[(df['game'] == 'Vindictus') & (df['action'] == 'play')]['id'].to_frame().drop_duplicates()
    df_VN['flg_VN'] = 1
    
    # merge flag back to master dataset
    df = df.merge(df_MS, on='id', how='left')\
           .merge(df_MA, on='id', how='left')\
           .merge(df_VN, on='id', how='left')\
           .fillna(0)
    
    # flag Nexon player
    df['flg_NX'] = df[['flg_MS', 'flg_MA', 'flg_VN']].max(axis=1)
    
    # player type
    df['player_type'] = np.where(df['flg_NX'] != 1, 'non_Nexon',
                        np.where((df['flg_MS'] == 1) & (df['flg_MA'] != 1) & (df['flg_VN'] != 1), 'MapleStory',
                        np.where((df['flg_MS'] != 1) & (df['flg_MA'] == 1) & (df['flg_VN'] != 1), 'Mabinogi',
                        np.where((df['flg_MS'] != 1) & (df['flg_MA'] != 1) & (df['flg_VN'] == 1), 'Vindictus', 'Nexon'))))
    
    return df

def find_top_n_game(df, top_n):
    '''
    This function returns the top_n game title in df.
    '''
    df['game_freq'] = df.groupby('game')['game'].transform('count')

    df = df[['game', 'game_freq']].drop_duplicates()\
                                  .sort_values('game_freq', ascending=False)\
                                  .reset_index(drop=True)
    top_n_game = df['game'][:top_n].tolist()
    top_n_game.sort()
    return top_n_game

def add_genre(df):
    '''
    This function adds genre to master dataframe
    '''    
    # get lsit of game
    df_game = df['game'].to_frame()\
                        .drop_duplicates()\
                        .sort_values(by='game')\
                        .reset_index(drop=True)

    # generate random vector of numbers
    df_game['index'] = np.random.randint(1, 6, size=len(df_game))

    # create a matching table
    genre = [[1, 'MMORPG'], [2, 'MOBA'], [3, 'FPS'], [4, 'RTS'], [5, 'Action']]
    df_genre = pd.DataFrame(genre, columns=['index', 'genre'])

    # merge game list with genre
    df_game = df_game.merge(df_genre, on='index', how='inner')
    df_game = df_game[['game', 'genre']].reset_index(drop=True)

    # merge genre with master df
    df = df.merge(df_game, on='game')
    
    return df

def feature_generation(df_knn):
    '''
    This function generates features from master dataframe:
    1. play time by genre
    2. number of purchase by genre
    3. % of time played by genre
    4. % of purchase by genre
    '''
    # create play time by genre dataset
    df_play_time = df_knn[df_knn['action'] == 'play']\
                        .filter(items=['id', 'genre', 'value'])\
                        .groupby(['id', 'genre'], as_index=False)['value']\
                        .sum()\
                        .pivot(index='id', columns='genre', values='value')\
                        .fillna(0)\
                        .rename(columns={'Action':'pt_action',
                                         'FPS':'pt_fps',
                                         'MMORPG':'pt_mmo',
                                         'MOBA':'pt_moba',
                                         'RTS':'pt_rts'})      
    # create play time by genre dataset
    df_purchase = df_knn[df_knn['action'] == 'purchase']\
                        .filter(items=['id', 'genre', 'value'])\
                        .groupby(['id', 'genre'], as_index=False)['value']\
                        .sum()\
                        .pivot(index='id', columns='genre', values='value')\
                        .fillna(0)\
                        .rename(columns={'Action':'pr_action',
                                         'FPS':'pr_fps',
                                         'MMORPG':'pr_mmo',
                                         'MOBA':'pr_moba',
                                         'RTS':'pr_rts'}) 
    # % of time played by genre
    df_time_porp = df_play_time.apply(lambda x: x/x.sum(), axis=1)\
                               .rename(columns={'pt_action':'ppt_action',
                                                'pt_fps':'ppt_fps',
                                                'pt_mmo':'ppt_mmo',
                                                'pt_moba':'ppt_moba',
                                                'pt_rts':'ppt_rts'})
    # % of purchase by genre
    df_purch_porp = df_purchase.apply(lambda x: x/x.sum(), axis=1)\
                               .rename(columns={'pr_action':'ppr_action',
                                                'pr_fps':'ppr_fps',
                                                'pr_mmo':'ppr_mmo',
                                                'pr_moba':'ppr_moba',
                                                'pr_rts':'ppr_rts'})
    # combine all dataset
    df_features = df_play_time.merge(df_purchase, left_index=True, right_index=True)\
                              .merge(df_time_porp, left_index=True, right_index=True)\
                              .merge(df_purch_porp, left_index=True, right_index=True)
    # convert index to id
    df_features['id'] = df_features.index
    df_features = df_features.reset_index(drop=True)
    
    return df_features