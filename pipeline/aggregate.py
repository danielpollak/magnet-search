"""Aggregate all *_analysis.pickle files into data/manuscript/all_fourier_df.parquet.

Usage:
    python pipeline/aggregate.py
    python pipeline/aggregate.py --out data/manuscript/all_fourier_df.parquet
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_OUT = os.path.join(DATA_DIR, "manuscript", "all_fourier_df.parquet")

# Maps rec name → [date, ID, area, species, contingency]
ANNOT_D = {
    'mag10Hz_2022-04-21_18-17-43':  ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"],
    'mag1Hz_2022-04-21_18-06-30':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"],
    'mag3Hz_2022-04-21_18-19-59':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"],
    'mag5Hz_2022-04-21_18-12-27':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"],
    'mag8Hz_2022-04-21_18-14-58':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"],
    '3HZ_2022-06-21_15-01-35':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"],
    '4HZ_2022-06-21_15-03-26':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"],
    '5HZ_2022-06-21_15-06-07':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"],
    '7HZ_2022-06-21_15-07-27':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"],
    'mag10hz_2022-04-08_15-26-22':  ['20220408', 'V649♀', 'NCM', 'zebra finch', "mag"],
    'mag3hz_2022-04-08_15-31-11':   ['20220408', 'V649♀', 'NCM', 'zebra finch', "mag"],
    'mag5hz_2022-04-08_15-28-47':   ['20220408', 'V649♀', 'NCM', 'zebra finch', "mag"],
    '2nd site 1 Hz_2022-02-28_17-54-46':  ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    '2nd site 10 Hz_2022-02-28_17-53-18': ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    '2nd site 2 Hz_2022-02-28_17-48-17':  ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    '2nd site 5 Hz_2022-02-28_17-50-52':  ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    'mag5hz_2022-03-14_18-15-42':   ['20220314', 'G122♀', 'arcopallium', 'zebra finch', "mag"],
    '10 Hz_2022-02-28_16-21-40':    ['20220228_firstsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    '2 Hz_2022-02-28_16-25-47':     ['20220228_firstsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    '5Hz_2022-02-28_16-06-03':      ['20220228_firstsite', 'V649♀', 'NCM', 'zebra finch', "mag"],
    'mag3hz_2023-02-16_11-43-42':   ['20230216', 'Pk12L', 'above OT', 'Pigeon', "mag"],
    'mag8hz2ndtry_2023-02-16_11-49-12': ['20230216', 'Pk12L', 'above OT', 'Pigeon', "mag"],
    'pigeon_CB_Mag1_inclined_2023-02-21_12-12-00':           ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_Mag1_inclined_lightson_2023-02-21_12-19-00':  ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_Mag2_inclined_2023-02-21_12-21-00':           ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_Mag5_inclined_2023-02-21_12-22-00':           ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_Mag7_inclined_2023-02-21_12-24-00':           ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_Mag7_upright_2023-02-21_12-26-02':            ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_mag1_upright_2023-02-21_12-33-18':            ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_mag2_upright_2023-02-21_12-30-43':            ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'pigeon_CB_oddball_5percent_2023-02-21_13-36-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    'pigeon_CB_WN3D_2023-02-21_12-35-40':            ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    'mag2_inclined_2023-02-28_17-12-50':             ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag2_inclined_lightson_2023-02-28_17-14-38':    ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag2_lightson_2023-02-28_17-18-29':             ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag2real_lightson_2023-02-28_17-20-26':         ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag5_inclined_lightson_2023-02-28_17-16-04':    ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'oddball_2023-02-28_17-25-32': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    'WN_2023-02-28_17-39-32':      ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    '2023-04-13_15-08-42_W25R_Mag2': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-11-11_W25R_Mag3': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-13-11_W25R_Mag8': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-15-40_W25R_Mag5': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-19-07_W25R_Mag2_inclined':                    ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-22-55_W25R_Mag3_inclined':                    ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-25-47_W25R_Mag3_inclined_repositioned':       ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-27-50_W25R_Mag5_inclined_repositioned_redo':  ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-29-41_W25R_Mag8_inclined_repositioned':       ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "mag"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_0':   ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_45':  ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_90':  ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_135': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_180': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_225': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_270': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-49-48_W25R_visual_3Hz_315': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_0':   ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_45':  ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_90':  ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_135': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_180': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_225': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_270': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_16-00-46_W25R_visual_2Hz_315': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-34-21_W25R_WN_IndepCh_redo': ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_15-39-50_W25R_WN_SameCh':       ['20230413_firstsite', 'W25R', 'centrolateral', 'Pigeon', "positive control"],
    '2023-04-13_17-04-34_W25R_second_site_mag2_inclined': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-06-27_W25R_second_site_mag3_inclined': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-10-21_W25R_second_site_mag8_inclined': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-13-18_W25R_second_site_mag8':          ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-15-32_W25R_second_site_mag5':          ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-17-14_W25R_second_site_mag3':          ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-18-59_W25R_second_site_mag2':          ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "mag"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_0':   ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_45':  ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_90':  ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_135': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_180': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_225': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_270': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_315': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-21-57_W25R_second_site_WN_SamCh':    ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-42-25_W25R_second_site_WN_IndepChan': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz0':   ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz45':  ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz90':  ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz135': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz180': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz225': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz270': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz315': ['20230413_secondsite', 'W25R', 'centromedial', 'Pigeon', "positive control"],
    '2023-04-14_14-28-40_W25R_mag2':                ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-30-22_W25R_mag3':                ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-32-14_W25R_mag5':                ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-34-11_W25R_mag8':                ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-37-41_W25R_mag8_inclined_lightson': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-42-29_W25R_mag5_inclined_lightson': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-44-59_W25R_mag3_inclined_lightson': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-47-16_W25R_mag2_inclined_lightson': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-50-42_W25R_mag2_lightson':          ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_14-53-11_W25R_mag3_lightson':          ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "mag"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_0':   ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_45':  ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_90':  ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_135': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_180': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_225': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_270': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-10-38_W25R_visual_3Hz_315': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_0':   ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_45':  ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_90':  ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_135': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_180': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_225': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_270': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-20-44_W25R_visual_2Hz_315': ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_14-56-27_W25R_WN_IndepCh':  ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-14_15-01-52_W25R_WN_SameCh':   ['20230414_firstsite', 'W25R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_15-56-12_W1R_mag2':          ['20230415', 'W1R', 'HP', 'Pigeon', "mag"],
    '2023-04-15_15-57-55_W1R_mag3':          ['20230415', 'W1R', 'HP', 'Pigeon', "mag"],
    '2023-04-15_15-59-40_W1R_mag8':          ['20230415', 'W1R', 'HP', 'Pigeon', "mag"],
    '2023-04-15_16-01-59_W1R_mag8_inclined': ['20230415', 'W1R', 'HP', 'Pigeon', "mag"],
    '2023-04-15_16-03-43_W1R_mag3_inclined': ['20230415', 'W1R', 'HP', 'Pigeon', "mag"],
    '2023-04-15_16-06-20_W1R_mag2_inclined': ['20230415', 'W1R', 'HP', 'Pigeon', "mag"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_0':   ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_45':  ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_90':  ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_135': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_180': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_225': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_270': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-08-19_W1R_visual_3Hz_315': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_0':   ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_45':  ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_90':  ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_135': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_180': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_225': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_270': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-19-13_W1R_visual_2Hz_315': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    '2023-04-15_16-37-23_W1R_3D_WN_Samechan': ['20230415', 'W1R', 'HP', 'Pigeon', "positive control"],
    'mag10_2022-09-16_15-53-06': ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"],
    'mag3_2022-09-16_15-42-31':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"],
    'mag5_2022-09-16_15-46-10':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"],
    'mag7_2022-09-16_15-48-29':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"],
    'mag9_2022-09-16_15-50-36':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"],
    'bars4_2022-09-16_15-31-32_0':   ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_45':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_90':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_135': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_180': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_225': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_270': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'bars4_2022-09-16_15-31-32_315': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'WN_2022-09-16_14-58-30':        ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'Q117_npxl_2022-12-13_s02_magnetD7': ['20221213', 'Q117', 'thalamus', 'Quail', "mag"],
    'Q117_npxl_2022-12-14_s02_magnetD7': ['20221214', 'Q117', 'thalamus', 'Quail', "mag"],
    'Q134_11.01.24_s01':                  ['20240111', 'Q134', 'Nidopallium', 'Quail', "mag"],
    'Q146_npxl_2023-08-15_magnet1_g0':    ['20230815', 'Q146', 'Nidopallium', 'Quail', "mag"],
    'q148magnet19-12s2_g0':  ['20241219', 'Q148', 'Nidopallium', 'Quail', "mag"],
    'q148magnet19-12s2_g1':  ['20241219', 'Q148', 'Nidopallium', 'Quail', "mag"],
    'magnerNPX2bank24shanks_g0': ['20240625', 'QNPX2', 'Nidopallium', 'Quail', "mag"],
    'magnerNPX2_g0':             ['20240625', 'QNPX2', 'Nidopallium', 'Quail', "mag"],
    # Medaka GCaMP — rec = basename(session_path) + ".tif" (set in analysis stage)
    'fish3_8dpf_magneto_0.tif':    ['20230106', 'fish_1', 'wholebrain', 'medaka', "mag"],
    'fish3_8dpf_magneto_1.tif':    ['20230106', 'fish_1', 'wholebrain', 'medaka', "mag"],
    'fish3_8dpf_magneto_2.tif':    ['20230106', 'fish_1', 'wholebrain', 'medaka', "mag"],
    'fish3_8dpf_no_magneto_0.tif': ['20230106', 'fish_1', 'wholebrain', 'medaka', "positive control"],
    'fish3_8dpf_no_magneto_1.tif': ['20230106', 'fish_1', 'wholebrain', 'medaka', "positive control"],
    'fish3_8dpf_no_magneto_2.tif': ['20230106', 'fish_1', 'wholebrain', 'medaka', "positive control"],
    # Mouse (KyuHyunLee) — Intan RHD .mat recordings, area=SC, loaded from data/precomputed/
    'magnetic4hz_170928_031259': ['20230928', 'mouse1', 'SC', 'mouse', "mag"],
    'magnetic_170928_025824':    ['20230928', 'mouse1', 'SC', 'mouse', "mag"],
    # Owl (Gutfreund) — pre-computed coefficients, loaded from data/precomputed/
    'exp1': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp2': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp3': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp4': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp5': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
}

# Engert GCaMP — species/ID/area to be filled in once fish identities are confirmed;
# contingency classified by tiff label (magneto = mag, no_magneto/nostim/visual = control)
_ENGERT_TRANCHE1_SESSIONS = {
    "20220221": ("2022_02_21", None),
    "20220223": ("2022_02_23", None),
    "20220301": ("2022_03_01", None),
}
for _date, (_sess, _id) in _ENGERT_TRANCHE1_SESSIONS.items():
    for _label, _cont in [
        ("magnet", "mag"), ("visualmagnet", "mag"),
        ("visual", "positive control"), ("nostim", "positive control"),
        ("visual_a", "positive control"), ("visual_b", "positive control"),
        ("visualmagnet_a", "mag"), ("visualmagnet_b", "mag"),
    ]:
        _name = f"engert_{_date}_{_label}"
        ANNOT_D[_name] = [_date, _id or "unknown", "whole brain", "zebrafish", _cont]

for _fish_tag, _date in [
    ("20220914_fish2", "20220914"), ("20220915_fish1", "20220915"),
    ("20221001_fish1", "20221001"), ("20221001_fish2", "20221001"),
    ("20221002_fish1", "20221002"), ("20221002_fish2", "20221002"),
]:
    for _i in range(3):
        ANNOT_D[f"engert_{_fish_tag}_magneto_{_i}"]    = [_date, _fish_tag, "whole brain", "zebrafish", "mag"]
        ANNOT_D[f"engert_{_fish_tag}_no_magneto_{_i}"] = [_date, _fish_tag, "whole brain", "zebrafish", "positive control"]
    for _i in range(1, 4):
        ANNOT_D[f"engert_{_fish_tag}_magneto_{_i}"]    = [_date, _fish_tag, "whole brain", "zebrafish", "mag"]
        ANNOT_D[f"engert_{_fish_tag}_no_magneto_{_i}"] = [_date, _fish_tag, "whole brain", "zebrafish", "positive control"]
        ANNOT_D[f"engert_{_fish_tag}_no-magneto_{_i}"] = [_date, _fish_tag, "whole brain", "zebrafish", "positive control"]


def build_all_fourier_df(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*_analysis.pickle")))
    if not files:
        raise FileNotFoundError(f"No *_analysis.pickle files found in {data_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            if not isinstance(df, pd.DataFrame):
                print(f"  skip {os.path.basename(f)}: not a DataFrame")
                continue
            dfs.append(df)
        except Exception as exc:
            print(f"  skip {os.path.basename(f)}: {exc}")
    print(f"Loaded {len(dfs)} / {len(files)} pipeline pickles")

    # Also load pre-computed species pickles (mouse, owl — formats not portable to YAML pipeline)
    precomputed_dir = os.path.join(data_dir, "precomputed")
    precomputed_files = sorted(glob.glob(os.path.join(precomputed_dir, "*.pickle")))
    n_pre = 0
    for f in precomputed_files:
        try:
            df = pd.read_pickle(f)
            if not isinstance(df, pd.DataFrame):
                print(f"  skip precomputed/{os.path.basename(f)}: not a DataFrame")
                continue
            if "contingency" not in df.columns:
                # infer: owl control row gets "control" (will be dropped later); rest → "mag"
                df = df.copy()
                df["contingency"] = df["rec"].apply(
                    lambda r: "control" if str(r).lower() == "control" else "mag"
                )
            dfs.append(df)
            n_pre += 1
        except Exception as exc:
            print(f"  skip precomputed/{os.path.basename(f)}: {exc}")
    if n_pre:
        print(f"Loaded {n_pre} precomputed pickle(s) from {precomputed_dir}")

    all_df = pd.concat(dfs, ignore_index=True)

    # Apply annotations
    annot_df = (
        pd.DataFrame.from_dict(ANNOT_D, orient="index",
                               columns=["date", "ID", "area", "species", "contingency"])
        .rename_axis("rec")
        .reset_index()
        .set_index("rec")
    )
    for col in ["date", "ID", "area", "species", "contingency"]:
        mapped = all_df["rec"].map(annot_df[col])
        if col in all_df.columns:
            all_df[col] = all_df[col].replace("", np.nan).fillna(mapped)
        else:
            all_df[col] = mapped

    # Standardise area labels (mirrors old save_aggregated_data logic)
    area_str = all_df["area"].astype(str).str.lower()
    area_str = area_str.replace({
        r".*centr.*":      "HP",
        r".*craniotomy.*": "HP",
        r".*cerebellum.*": "CB",
        r".*ot.*":         "pallium",
    }, regex=True).replace({
        "hp":         "HP",
        "ncm":        "NCM",
        "sc":         "SC",
        "whole brain": "WB",
        "wholebrain":  "WB",
        "nan":         np.nan,
    })
    all_df["area"] = area_str

    # Drop legacy Owl control row (no magnetic stimulus)
    all_df = all_df.loc[~((all_df.species == "Owl") & (all_df.rec == "control"))]

    unannotated = all_df.loc[all_df["species"].isna(), "rec"].unique()
    if len(unannotated):
        print(f"  {len(unannotated)} rec(s) have no annotation — add to ANNOT_D in pipeline/aggregate.py:")
        for r in unannotated[:20]:
            print(f"    {r!r}")

    return all_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Aggregate analysis pickles → parquet")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Directory containing *_analysis.pickle files (default: {DATA_DIR})")
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} ...")
    df = build_all_fourier_df(args.data_dir)
    print(f"Total rows: {len(df):,}   columns: {list(df.columns)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
