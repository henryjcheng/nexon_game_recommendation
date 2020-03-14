'''
This program contains functions that are ran in Jupyter Notebook.
The purpose is to clean up Notebook so they are more readable.
'''

def eda_phase1(df):
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