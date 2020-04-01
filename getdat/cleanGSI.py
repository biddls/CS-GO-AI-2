import time


def main(data):
    player = data.get('player')
    if player.get('observer_slot') == 1:
        map = data.get('map')

        round = data.get('round')

        try:
            round.pop('win_team')
        except:
            pass

        if len(round) < 2:
            round = [round.get('phase'), 'not planted']

        else:
            round = [round.get('phase'), round.get('bomb')]

        map = [map.get('team_ct').get('score'), map.get('team_t').get('score'), round[0], round[1]]

        stats = player.get('match_stats')
        stats = [stats.get('kills'), stats.get('assists'), stats.get('deaths'), stats.get('mvps'), stats.get('score')]

        state = player.get('state')
        if state.get('round_killshs') == None:
            HSs = 0
        else:
            HSs = state.get('round_killshs')
        state = [state.get('health'), state.get('flashed'), state.get('smoked'), state.get('burning'),
                 state.get('round_kills'), HSs]

        player = [player.get('team')]

        player += state + stats

        data = map + player

        path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\game data\\data.txt'
        dat = open(path, 'a+')
        string = str(data[0])

        for x in data[1:]:
            string += ', ' + str(x)

        dat.write(str(time.time()) + string + '\r')

        dat.close()

        key = ['ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked',
               'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score'] #and time idk why i have that may remove

        prnt = []
        for x in range(len(key)):
            prnt.append([key[x], data[x]])

        print(prnt)
    else:
        print('Oberserving/ dead')
