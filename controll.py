#!-*-coding:utf-8-*-
import os
import sys
sys.path.append( os.path.dirname(__file__ ) )
import lib
import json
import shutil
import warnings
import pandas as pd
from lib.tools.logger import *

warnings.simplefilter("ignore")
__all__ = ["backborn"]

class backborn:
    """
    様々な機能をコントロールする箱
    --------------------------------
    input :
        - data_path : 分析用データの場所 <string>
        - setting_path : 設定ファイルの場所 <string>
        - mode : 実行モード <integer>
    --------------------------------
    """
    def __init__( self , work_dir , mode=0 ):
        self.data_path = work_dir + "datas/"
        self.setting_path = work_dir + "settings/"
        self.models_path = work_dir + "models/"
        self.simurates_path = work_dir + "simurates/"
        self.mode = mode

    def set( self ):
        #必要なフォルダを作成する
        if os.path.exists( self.data_path ) is not True:
            os.makedirs( self.data_path )
            os.makedirs( self.models_path )
            os.makedirs( self.data_path + "originals" )
            os.makedirs( self.data_path + "shaped" )
            os.makedirs( self.data_path + "statistics" )
            os.makedirs( self.data_path + "simurates" )
            message( "mkdir {}".format( self.data_path ) , mode=self.mode )

        if os.path.exists( self.setting_path ) is not True:
            shutil.copytree( "{}/templates".format(os.path.dirname( __file__) ) , self.setting_path )
            message( "set template files to {}".format( self.setting_path ) , mode=self.mode )
            message( "please set your data to {}".format( self.data_path ) )
            message( "please set information to {}".format( self.setting_path ) )


        self.all_files = self.__get_all_files()

        return True

    def simuration( self ):
        exec_infos = lib.json2dict( self.setting_path + "simurate.json" )
        for model , options in exec_infos.items():
            exec = eval( "lib.{}".format( model ) )
            for method , details in options.items():
                exec = exec( details , self.simurates_path )
                eval( "exec.{}".format( method ) )()

    def predict( self ):
        """
        import pickle as pkl

        filename = "models/decisiontree_model_決済ステータスID_契約時年齢_契約時年収_金融機関_独身・既婚.sav"
        model = pkl.load( open( filename , "rb" ) )
        print( model )
        """
        pass

    def learn( self ):
        exec_infos = lib.json2dict( self.setting_path + "learn.json" )
        exec_array = self.__get_exec( exec_infos=exec_infos )
        for exec in exec_array:
            self.__learn_dump( exec=exec , mode=self.mode )

    def csv( self ):
        #######################################
        #             データマージ              #
        #######################################
        execed_array = []
        exec_infos = lib.json2dict( self.setting_path + "csv_controll.json" )
        exec_array = self.__get_exec( exec_infos=exec_infos )
        while len( exec_array ) != len( execed_array ):
            for exec in exec_array:
                if list( exec.keys() )[0] not in execed_array:
                    self.__csv_dump( exec=exec , mode=self.mode )
                    execed_array += list( exec.keys() )

            exec_array = self.__get_exec( exec_infos=exec_infos )


    def describe( self ):
        #######################################
        #              記述統計                #
        #######################################
        execed_array = []
        exec_infos = lib.json2dict( self.setting_path + "graphs.json" )
        exec_array = self.__get_exec( exec_infos=exec_infos )
        while len( exec_array ) != len( execed_array ):
            for exec in exec_array:
                if list( exec.keys() )[0] not in execed_array:
                    self.__graph_dump( exec=exec , mode=self.mode )
                    execed_array += list( exec.keys() )

            exec_array = self.__get_exec( exec_infos=exec_infos )


    def aggregate( self ):
        #######################################
        #                集計                 #
        #######################################
        exec_infos = lib.json2dict( self.setting_path + "aggregate.json" )
        exec_array = self.__get_exec( exec_infos=exec_infos )
        for exec in exec_array:
            self.__aggregate_dump( exec=exec , mode=self.mode )

        #新しく生成されたファイルの情報を取得
        new_array = self.__get_exec( exec_infos=exec_infos )
        for exec in exec_array:
            new_array.remove( exec )

        #実行
        for exec in new_array:
            self.__aggregate_dump( exec=exec , mode=self.mode )


    def __get_exec( self , exec_infos ):
        #######################################
        #           集計などを実行する           #
        #######################################
        #初期化
        exec_array = []
        exec_file_array = list( exec_infos.keys() )
        self.all_files = self.__get_all_files()
        #データファイルを取得し、jsonファイルに情報があれば実行をする
        for file_name in self.all_files:
            exec_dict = self.__command_change( exec_infos=exec_infos ,\
             file_array=exec_file_array , file_name=file_name )

            if len( list( exec_dict.keys() ) ) > 0:
                    exec_array.append( exec_dict )


        return exec_array

    """
    Exe Local Methods
    """
    def __learn_dump( self , exec , mode=0 ):
        #######################################
        #           推定結果を出力する           #
        #######################################
        for path , dict in exec.items():
            df = pd.read_csv( path )
            for learn_format , vals in dict.items():
                print( "\n" )
                system( ":::::{}:::::".format( learn_format.split("_")[0].upper() ) )
                learn_format = learn_format.split("_")[0]
                exe = eval( "lib.{}".format( learn_format ) )( df , vals , path )
                model = exe.learn()
                exe.dump( model=model )
                acc = exe.accuracy( model=model )
                system( "Learning {} -> N :: {} train spec :: {} test spec :: {}"\
                .format( learn_format , acc["N"] , acc["train"] , acc["test"] ) )



    def __aggregate_dump( self , exec , mode=0 ):
        #######################################
        #           集計結果を出力する           #
        #######################################
        for path , dict in exec.items():
            df = pd.read_csv( path )
            for aggregate_format , vals in dict.items():
                num = aggregate_format.split("_")[1]
                aggregate_format = aggregate_format.split("_")[0] #連番を除去
                exe = eval( "lib.{}".format( aggregate_format ) )( df , vals )
                aggregate_obj = exe.dump()
                save_path = os.path.dirname( path )
                save_path = "{}/{}/{}_{}_{}.csv".format( save_path \
                , os.path.basename( path ).split(".")[0], aggregate_format ,\
                 json.loads( vals["agg"] )[1] , num )
                save_path = save_path.replace("shaped","statistics")
                save_path = save_path.replace("originals","statistics")

                if os.path.exists( os.path.dirname( save_path ) ) is not True:
                    os.makedirs( os.path.dirname( save_path ) )
                    message( "mkdir {}".format( save_path ) , mode=self.mode )

                self.__mode_change( mode=mode , obj=aggregate_obj.to_csv ,\
                 save_path=save_path )


    def __graph_dump( self , exec , mode=0 ):
        #######################################
        #            グラフを出力する            #
        #######################################
        for path , dict in exec.items():
            df = pd.read_csv( path )
            for graph_format , vals in dict.items():
                # 設定ファイルにあるメソッド名を動的に実行
                graph_format = graph_format.split("_")[0] # 連番を除去
                exe = eval( "lib.{}".format( graph_format ) )( df , vals )
                figure_obj = exe.dump()
                save_path = os.path.dirname( path.replace("datas","graphs") )
                if os.path.exists( save_path ) is not True:
                    os.makedirs( save_path  )
                    message( "mkdir {}".format( save_path ) , mode=self.mode )

                save_path = "{}/{}_{}_{}_{}.png".format( save_path ,\
                 os.path.basename( path ).split(".")[0] ,\
                  graph_format , json.loads( vals["y"] )[0] , json.loads( vals["x"] )[0] )

                self.__mode_change( mode=mode , obj=figure_obj.savefig ,\
                 save_path=save_path )


    def __csv_dump( self , exec , mode=0 ):
        #結合するcsvは必ず同じフォルダに格納する
        for path , dict in exec.items():
            save_path = os.path.basename( path ).split(".")[0]
            df = pd.read_csv( path )
            for csv_mode , vals in dict.items():
                csv_mode = csv_mode.split("_")[0] #連番を除去
                exe = eval( "lib.{}".format( csv_mode ) )( path , vals , df )
                output_csvinfo_dict = exe.dump()
                if output_csvinfo_dict["csv_name"] is not False:
                    df = output_csvinfo_dict["dataframe"]
                    save_path += "_{}".format( csv_mode )

            save_file_name = "{}/{}.csv".format( output_csvinfo_dict["save_path"] , save_path )
            if df is not None:
                self.__mode_change( mode=mode , obj=df.to_csv ,\
                 save_path=save_file_name )

    """
    Another Local Methods
    """
    def __listdir( self , path ):
        return [ filename for filename in os.listdir( path ) if not filename.startswith('.') ]


    def __command_change( self , exec_infos , file_array , file_name ):
        #######################################
        #     深い階層ならさらにファイルを展開     #
        #######################################
        dict = {}
        if os.path.isdir( file_name ):
            for file in self.__listdir( file_name ):
                file_path = file_name + "/" + file
                if file in file_array:
                    file = os.path.basename( file )
                    dict[file_path] = exec_infos[file]
        else:
            file = os.path.basename( file_name )
            if file in file_array:
                dict[file_name] = exec_infos[file]

        return dict

    def __mode_change( self , mode , obj , save_path ):
        if mode == 0:
            return False
        else:
            dirname = os.path.dirname( save_path )
            if os.path.exists( dirname ) is not True:
                os.makedirs( dirname )
            obj( save_path )
            message( "save figure in {}".format( save_path ) , mode=mode )


    def __get_all_files( self ):
        files = []
        for folder in self.__listdir( self.data_path ):
            for child_folder in self.__listdir( self.data_path + folder ):
                files.append( "{}/{}".format( self.data_path + folder , child_folder ) )
        return files
