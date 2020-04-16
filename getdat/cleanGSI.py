import time
import traceback

def clean(data):
    player = data.get('player')
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

    map = [map.get('team_ct').get('score'), map.get('team_t').get('score'), round[0], round[1]]#info about the map

    stats = player.get('match_stats')
    stats = [stats.get('kills'), stats.get('assists'), stats.get('deaths'), stats.get('mvps'), stats.get('score')]#players performance

    state = player.get('state')
    if state.get('round_killshs') == None:#if HS have been made it gets them here
        HSs = 0
    else:
        HSs = state.get('round_killshs')

    state = [state.get('health'), state.get('flashed'), state.get('smoked'), state.get('burning'),
             state.get('round_kills'), HSs]#state of character

    player = [player.get('team')]

    player += state + stats
    data = map + player#gets all data filtered here

    get = ['kills', 'deaths']

    both = dict(zip(key,data))#easy way of getting whats wanted from the CS data output
    use = []
    for item in get:
        use.append(both[item])

    path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt'

    comp = None
    dat = open(path, 'r')
    comp2 = dat.read().split('\n')
    comp2.pop()
    for x in comp2:
        comp = x#gets most recent kills
    dat.close()

    comp2 = ", ".join(str(x) for x in use)#parses used data into fomrat to be saved in

    if comp != comp2:#if the ouput is diffrent (somthing has changed) then it saves the new info
        dat = open(path, 'a+')
        dat.write(comp2 + '\r')
        dat.close()

    prnt = []#formats data to output nicely
    for x in range(len(key)):
        prnt.append([key[x], data[x]])

    #print(prnt)

key = ['ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked',
           'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps',
           'score']

def main(data):
    try:
        if data.get('player').get('state').get('health') > 0:#if alive (not spectating etc stuff)
            clean(data)

        """else:
            path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt'
            dat = open(path, 'a+')

            clean(data)
            dat.write('DIED\r')

            dat.close()"""
    except:
        open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt',
             "w+").close()