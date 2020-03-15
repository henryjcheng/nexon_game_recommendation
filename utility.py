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

'''
Here we will create a dictionary for games, which includes game titile and its genre
'''
game_dictionary = { '7 Days to Die': 'surivor horror',
                    'APB Reloaded': '',
                    'ARK Survival Evolved': ,
                    'AdVenture Capitalist': ,
                    'Age of Empires II HD Edition': ,
                    'Alan Wake': ,
                    'Alien Swarm': ,
                    'Amnesia The Dark Descent': ,
                    'Archeblade': ,
                    'Arma 2': ,
                    'Arma 2 Operation Arrowhead': ,
                    'Arma 2 Operation Arrowhead Beta (Obsolete)': ,
                    'Arma 3': ,
                    'BLOCKADE 3D': ,
                    'Bastion': ,
                    'Batman Arkham Asylum GOTY Edition': ,
                    'Batman Arkham City GOTY': ,
                    'Batman Arkham Origins': ,
                    'BattleBlock Theater': ,
                    'Battlefield Bad Company 2': ,
                    'BioShock': ,
                    'BioShock 2': ,
                    'BioShock Infinite': ,
                    'Blacklight Retribution': ,
                    'Borderlands': ,
                    'Borderlands 2': ,
                    'Burnout Paradise The Ultimate Box': ,
                    'Call of Duty Black Ops': ,
                    'Call of Duty Black Ops - Multiplayer': ,
                    'Call of Duty Black Ops II': ,
                    'Call of Duty Black Ops II - Multiplayer': ,
                    'Call of Duty Black Ops II - Zombies': ,
                    'Call of Duty Modern Warfare 2': ,
                    'Call of Duty Modern Warfare 2 - Multiplayer': ,
                    'Call of Duty Modern Warfare 3': ,
                    'Call of Duty Modern Warfare 3 - Multiplayer': ,
                    'Castle Crashers': ,
                    'Chivalry Medieval Warfare': ,
                    'Cities Skylines': ,
                    'Clicker Heroes': ,
                    'Company of Heroes': ,
                    'Company of Heroes (New Steam Version)': ,
                    'Company of Heroes 2': ,
                    'Counter-Strike': ,
                    'Counter-Strike Condition Zero': ,
                    'Counter-Strike Condition Zero Deleted Scenes': ,
                    'Counter-Strike Global Offensive': ,
                    'Counter-Strike Nexon Zombies': ,
                    'Counter-Strike Source': ,
                    'Cry of Fear': ,
                    'Crysis 2 Maximum Edition': ,
                    'DC Universe Online': ,
                    'Dark Souls Prepare to Die Edition': ,
                    'Darksiders': ,
                    'Day of Defeat': ,
                    'Day of Defeat Source': ,
                    'DayZ': ,
                    'Dead Island': ,
                    'Dead Island Epidemic': ,
                    'Dead Space': ,
                    'Deathmatch Classic': ,
                    'Defiance': ,
                    'Deus Ex Human Revolution': ,
                    'Dirty Bomb': ,
                    'Dishonored': ,
                    'Don\'t Starve': ,
                    'Don\'t Starve Together Beta': ,
                    'Dota 2': ,
                    'Dungeon Defenders': ,
                    'Dungeon Defenders II': ,
                    'Empire Total War': ,
                    'Euro Truck Simulator 2': ,
                    'FTL Faster Than Light': ,
                    'Fallout 3 - Game of the Year Edition': ,
                    'Fallout 4': ,
                    'Fallout New Vegas': ,
                    'Fallout New Vegas Dead Money': ,
                    'Fallout New Vegas Honest Hearts': ,
                    'Far Cry 3': ,
                    'Firefall': ,
                    'Fistful of Frags': ,
                    'Football Manager 2013': ,
                    'FreeStyle2 Street Basketball': ,
                    'Garry\'s Mod': ,
                    'Goat Simulator': ,
                    'Gotham City Impostors Free To Play': ,
                    'Grand Theft Auto Episodes from Liberty City': ,
                    'Grand Theft Auto IV': ,
                    'Grand Theft Auto San Andreas': ,
                    'Grand Theft Auto V': ,
                    'GunZ 2 The Second Duel': ,
                    'H1Z1': ,
                    'HAWKEN': ,
                    'Half-Life': ,
                    'Half-Life 2': ,
                    'Half-Life 2 Deathmatch': ,
                    'Half-Life 2 Episode One': ,
                    'Half-Life 2 Episode Two': ,
                    'Half-Life 2 Lost Coast': ,
                    'Half-Life Blue Shift': ,
                    'Half-Life Deathmatch Source': ,
                    'Half-Life Opposing Force': ,
                    'Half-Life Source': ,
                    'Heroes & Generals': ,
                    'Hitman Absolution': ,
                    'Hotline Miami': ,
                    'Insurgency': ,
                    'Just Cause 2': ,
                    'Killing Floor': ,
                    'Killing Floor Mod Defence Alliance 2': ,
                    'LIMBO': ,
                    'Left 4 Dead': ,
                    'Left 4 Dead 2': ,
                    'Loadout': ,
                    'Mafia II': ,
                    'Magicka': ,
                    'Magicka Wizard Wars': ,
                    'Marvel Heroes 2015': ,
                    'Max Payne 3': ,
                    'Metro 2033': ,
                    'Middle-earth Shadow of Mordor': ,
                    'Mirror\'s Edge': ,
                    'Mount & Blade Warband': ,
                    'Napoleon Total War': ,
                    'Natural Selection 2': ,
                    'Neverwinter': ,
                    'No More Room in Hell': ,
                    'Nosgoth': ,
                    'ORION Prelude': ,
                    'Orcs Must Die! 2': ,
                    'PAYDAY 2': ,
                    'PAYDAY The Heist': ,
                    'Patch testing for Chivalry': ,
                    'Path of Exile': ,
                    'PlanetSide 2': ,
                    'Portal': ,
                    'Portal 2': ,
                    'Prison Architect': ,
                    'Quake Live': ,
                    'RIFT': ,
                    'RaceRoom Racing Experience': ,
                    'Realm of the Mad God': ,
                    'Red Orchestra 2 Heroes of Stalingrad - Single Player': ,
                    'Ricochet': ,
                    'Rising Storm/Red Orchestra 2 Multiplayer': ,
                    'Robocraft': ,
                    'Rocket League': ,
                    'Rust': ,
                    'SMITE': ,
                    'Saints Row 2': ,
                    'Saints Row IV': ,
                    'Saints Row The Third': ,
                    'Serious Sam 3 BFE': ,
                    'Serious Sam HD The Second Encounter': ,
                    'Sid Meier\'s Civilization V': ,
                    'Sid Meier\'s Civilization V Brave New World': ,
                    'Skyrim High Resolution Texture Pack': ,
                    'Sniper Elite V2': ,
                    'South Park The Stick of Truth': ,
                    'Space Engineers': ,
                    'Spiral Knights': ,
                    'Star Wars - Battlefront II': ,
                    'Star Wars Knights of the Old Republic': ,
                    'Starbound': ,
                    'Super Meat Boy': ,
                    'Surgeon Simulator': ,
                    'Survarium': ,
                    'TERA': ,
                    'Tactical Intervention': ,
                    'Team Fortress 2': ,
                    'Team Fortress Classic': ,
                    'Terraria': ,
                    'The Binding of Isaac': ,
                    'The Elder Scrolls V Skyrim': ,
                    'The Elder Scrolls V Skyrim - Dawnguard': ,
                    'The Elder Scrolls V Skyrim - Dragonborn': ,
                    'The Elder Scrolls V Skyrim - Hearthfire': ,
                    'The Expendabros': ,
                    'The Forest': ,
                    'The Mighty Quest For Epic Loot': ,
                    'The Stanley Parable': ,
                    'The Walking Dead': ,
                    'The Witcher 2 Assassins of Kings Enhanced Edition': ,
                    'The Witcher Enhanced Edition': ,
                    'Tom Clancy\'s Ghost Recon Phantoms - EU': ,
                    'Tom Clancy\'s Ghost Recon Phantoms - NA': ,
                    'Tomb Raider': ,
                    'Torchlight II': ,
                    'Toribash': ,
                    'Total War ROME II - Emperor Edition': ,
                    'Total War SHOGUN 2': ,
                    'Trine 2': ,
                    'Trove': ,
                    'Unturned': ,
                    'War Thunder': ,
                    'Warface': ,
                    'Warframe': ,
                    'Warhammer 40,000 Dawn of War II': ,
                    'XCOM Enemy Unknown': ,
                    'theHunter': 
                    }