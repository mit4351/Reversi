# -*- coding: utf-8 -*-

import wx
import ReverseBoard
import Player
import ReverseCommon
import datetime

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self._SenteColor = ReverseCommon.BLACK
        self.SetSize((550, 550))
        self.SetTitle("My Game")
        self.Centre()
        self.CreateStatusBar()
        self.SetMyMenuBar()
        self.InitUI()
        self.setPlayer()
        self.Show()

    def InitUI(self, pRow=8, pCol=8):
        mainPanel = wx.Panel(self)
        mainPanel.SetBackgroundColour("LIGHT BLUE")

        grid = wx.GridSizer(pRow, pCol,2,2)
        count = 1
        self.Stones = {}
        self.Labels = {}
        for r in range(pRow):
            for c in range (pCol):
                wPanel = wx.Panel(mainPanel,count)
                wPanel.SetBackgroundColour(wx.GREEN)
                # 右クリック時のイベント登録
                wPanel.Bind(wx.EVT_LEFT_DOWN, self.StoneClick)
                grid.Add(wPanel,1,flag=wx.EXPAND)
                wlabel = str(count)
                #wlabel = "＊"
                self.Labels[str(count)] = wx.StaticText(wPanel, wx.ID_ANY, wlabel, style=wx.TE_CENTER)
                self.Labels[str(count)].SetForegroundColour(wx.GREEN)
                self.Stones[str(count)] = wPanel
                layout =  wx.BoxSizer(wx.VERTICAL)
                layout.Add(wx.StaticText(wPanel, -1, ''))
                layout.Add(self.Labels[str(count)], proportion=1,flag=wx.GROW)
                layout.Add(wx.StaticText(wPanel, -1, ''))
                wPanel.SetSizer(layout)

                count += 1

        mainPanel.SetSizer(grid)

    def setStoneColor(self):
        self.StartFlg = True
        Index_GREEN = ReverseCommon.get_index(self._reverse_board.stone_status, ReverseCommon.NONE)
        for i in Index_GREEN:
            self.Stones[str(i)].SetBackgroundColour(wx.GREEN)
            self.Labels[str(i)].SetForegroundColour(wx.GREEN)

        Index_BLACK = ReverseCommon.get_index(self._reverse_board.stone_status, ReverseCommon.BLACK)
        for i in Index_BLACK:
            self.Stones[str(i)].SetBackgroundColour(wx.BLACK)
            self.Labels[str(i)].SetForegroundColour(wx.BLACK)

        Index_WHITE = ReverseCommon.get_index(self._reverse_board.stone_status, ReverseCommon.WHITE)
        for i in Index_WHITE:
            self.Stones[str(i)].SetBackgroundColour(wx.WHITE)
            self.Labels[str(i)].SetForegroundColour(wx.WHITE)

        # 石を置ける全候補地
        all_candidates = ReverseCommon.get_puttable_points(self._reverse_board.stone_status, self._reverse_board.CurrentColor)
        for pos in all_candidates:
            self.Labels[str(ReverseCommon.PointToIndex(pos))].SetForegroundColour(wx.RED)

        self.Refresh()

        BLACK_score = ReverseCommon.get_score(self._reverse_board.stone_status, ReverseCommon.BLACK)
        WHITE_score = ReverseCommon.get_score(self._reverse_board.stone_status, ReverseCommon.WHITE)
        if self._playerAI.color == ReverseCommon.BLACK:
            kuro = "AI"
            siro = "貴方"
        else:
            kuro = "貴方"
            siro = "AI"

        wMessage = '黒('  + kuro + '):' + str(BLACK_score) + '  vs  ' +  str(WHITE_score) + ':(' + siro + ')白'
        self.SetStatusText(wMessage)

    def setPlayer(self):
        # プレイヤー
        #self._playerAI = Player.RandomAi(not(self._SenteColor))
        #self._playerAI = Player.NextStoneMaxAi(not(self._SenteColor))
        #self._playerAI = Player.RandomAiKnowGoodMove(not(self._SenteColor))
        #self._playerAI = Player.DeepQNetWork(not(self._SenteColor),"AI_Deep.pkl")
        #self._playerAI = Player.DNet(not(self._SenteColor),"AI_DQN.pkl")
        self._playerAI = Player.QNetWork(not(self._SenteColor),"AI_Deep.pkl","AI_Deep2.pkl")

        self._reverse_board = ReverseBoard.ReverseBoard()
        #ReverseCommon.print_board(self._reverse_board.board)
        self.setStoneColor()

        if self._SenteColor == ReverseCommon.WHITE:
            self.AI_teban()

    def StoneClick(self, event):
        if not self.StartFlg:
            return

        click = event.GetEventObject()  # クリックされたのはどのオブジェクトか
        # 置けなくなったら終了
        if self._reverse_board.is_game_set():
            self.EndProc()
            return

        # 置ける場所か確認
        if  not ReverseCommon.isPuttable_points(self._reverse_board, ReverseCommon.IndexToPoint(click.GetId())):
            return

        #人間手番反映
        self._reverse_board.put_stone2(ReverseCommon.IndexToPoint(click.GetId()))
        self.setStoneColor()
        #print("黒番>>" + str(click.GetId()) + "::" + str(ReverseCommon.IndexToPoint(click.GetId())))

        # 置けなくなったら終了
        if self._reverse_board.is_game_set():
            self.EndProc()
            return

        # AI手番置けるか確認(置けない場合再度人間手番)
        enemy = not(self._playerAI.color)
        if  self._reverse_board.CurrentColor == enemy :
            print("<<AI手番スキップ>>")
            wx.MessageBox("AI手番スキップ", '情報')
            return

        self.AI_teban()

    def AI_teban(self):
        while True:
            next_move = self._playerAI.next_move(self._reverse_board)
            self._reverse_board.put_stone2(next_move)
            self.setStoneColor()
            #print("白番>>" + str(ReverseCommon.PointToIndex(next_move)) + "::" + str(next_move))
            #print(ReverseCommon.boardToKey(self._reverse_board.stone_status))
            #self._playerAI.update_network()

            # 置けなくなったら終了
            if self._reverse_board.is_game_set():
                self.EndProc()
                return

            # 人間手番置けるか確認(置けない場合再度AI手番)
            if  self._reverse_board.CurrentColor == self._playerAI.color:
                print("<<貴方手番スキップ>>")
                wx.MessageBox("貴方手番スキップ", '情報')
            else:
                return

    def EndProc(self):
        # スコアーを集計
        BLACK_score = ReverseCommon.get_score(self._reverse_board.stone_status, ReverseCommon.BLACK)
        WHITE_score = ReverseCommon.get_score(self._reverse_board.stone_status, ReverseCommon.WHITE)
        if self._playerAI.color == ReverseCommon.BLACK:
            kuro = "AI"
            siro = "貴方"
        else:
            kuro = "貴方"
            siro = "AI"

        wMessage = ' '*5 +  '黒('  + kuro + '):' + str(BLACK_score) + '  vs  ' +  str(WHITE_score) + ':(' + siro + ')白' + ' '*5

        self._playerAI.update_network(ReverseCommon.get_score(self._reverse_board.stone_status, self._playerAI.color) - 32)

        if BLACK_score == WHITE_score :
            wMessage += "引き分け"
        elif BLACK_score > WHITE_score :
            wMessage += "黒(" + kuro + ")の勝"
        else:
            wMessage += "白(" + siro + ")の勝"

        wMessage += "　　もう一度?"
        #wx.MessageBox(wMessage + wResult, '終了♪')
        dialog = wx.MessageDialog(None, wMessage, '終了♪',style=wx.YES_NO | wx.ICON_INFORMATION)
        res = dialog.ShowModal()
        if res == wx.ID_YES:
            self._reverse_board = ReverseBoard.ReverseBoard()
            self.setStoneColor()
            print("Yes")
        elif res == wx.ID_NO:
            self.StartFlg = False
            print("No")

    def selectMenu(self, event):
        #self.SetStatusText("MenuSelected! " + str(event.GetId()))
        if(event.GetId() == 2):
            self.Close(True)
        if(event.GetId() == 5):
            self.setPlayer()
            #self._reverse_board = ReverseBoard.ReverseBoard()
            #self.setStoneColor()
        if(event.GetId() == 1):
            print("保存")
            self._playerAI.save_network("AI_QNetWork.pkl")
        if(event.GetId() == 7):
            self._SenteColor = not(self._SenteColor)
            self.setPlayer()

    def SetMyMenuBar(self):
        menu_file = wx.Menu()
        menu_file.Append(5, "&初めから\tCtrl+S")
        menu_file.Append(7, "&手番交代\tCtrl+R")
        menu_file.Append(6, u"履歴")
        menu_file.Append(2, '&終了\tCtrl+Q')
        menu_edit = wx.Menu()
        menu_edit.Append(3, '&設定\tCtrl+E')
        menu_edit.Append(1, u"保存")
        menu_edit.Append(4, u"学習")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu_file, u"ゲーム")
        menu_bar.Append(menu_edit, u"設定")
        self.Bind(wx.EVT_MENU,self.selectMenu)
        self.SetMenuBar(menu_bar)

def DisplayBorde():
    app = wx.App()
    MyFrame(None).Show(True)
    app.MainLoop()

if __name__ == '__main__':
    DisplayBorde()
