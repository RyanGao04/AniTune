#!/usr/bin/env python3
"""
Comprehensive script to find English/Japanese names for Chinese anime character names.
Uses known anime series mappings and character databases.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

# Known anime series and their character mappings
ANIME_SERIES_MAPPINGS = {
    # Slam Dunk (灌篮高手)
    "灌篮高手": {
        "安西光义": {"english": "Anzai Kōzō", "japanese": "安西光義"},
        "流川枫": {"english": "Rukawa Kaede", "japanese": "流川楓"},
        "木暮禅次朗": {"english": "Kogure Zenjirō", "japanese": "木暮禅次郎"},
    },
    
    # Naruto (火影忍者)
    "火影忍者": {
        "日向宁次": {"english": "Hyuga Neji", "japanese": "日向ネジ"},
    },
    
    # One Piece (海贼王)
    "海贼王": {
        "文斯莫克·尼治": {"english": "Vinsmoke Niji", "japanese": "ヴィンスモーク・ニジ"},
        "赛尼奥尔·皮克": {"english": "Senor Pink", "japanese": "セニョール・ピンク"},
        "狐火锦卫门": {"english": "Kin'emon", "japanese": "錦えもん"},
        "老鼠上校": {"english": "Nezumi", "japanese": "ネズミ"},
    },
    
    # Food Wars (食戟之灵)
    "食戟之灵": {
        "塔克米·阿尔迪尼": {"english": "Takumi Aldini", "japanese": "タクミ・アルディーニ"},
        "水户郁魅": {"english": "Mito Ikumi", "japanese": "水戸郁魅"},
    },
    
    # Sakurasou (樱花庄的宠物女孩)
    "樱花庄的宠物女孩": {
        "神田空太": {"english": "Kanda Sorata", "japanese": "神田空太"},
    },
    
    # Cardcaptor Sakura (魔卡少女樱)
    "魔卡少女樱": {
        "李莓铃": {"english": "Li Meiling", "japanese": "李苺鈴"},
    },
    
    # Detective Conan (名侦探柯南)
    "名侦探柯南": {
        "大木绫子": {"english": "Omaki Ayako", "japanese": "大木綾子"},
        "原佳明": {"english": "Hara Yoshimasa", "japanese": "原佳明"},
        "弓长警部": {"english": "Takei Keibu", "japanese": "弓長警部"},
    },
    
    # Dragon Ball (龙珠)
    "龙珠": {
        "雅木茶": {"english": "Yamcha", "japanese": "ヤムチャ"},
    },
    
    # Ultraman series
    "奥特曼": {
        "布鲁奥特曼": {"english": "Ultraman Blu", "japanese": "ウルトラマンブル"},
        "罗索奥特曼": {"english": "Ultraman Rosso", "japanese": "ウルトラマンルーブ"},
        "奥特之王": {"english": "Ultraman King", "japanese": "ウルトラマンキング"},
        "究极赛罗": {"english": "Ultraman Zero Beyond", "japanese": "ウルトラマンゼロビヨンド"},
    },
    
    # Kamen Rider (假面骑士)
    "假面骑士": {
        "假面骑士王蛇": {"english": "Kamen Rider Ouja", "japanese": "仮面ライダー王蛇"},
    },
    
    # Pokemon (精灵宝可梦)
    "精灵宝可梦": {
        "快泳蛙": {"english": "Poliwrath", "japanese": "ニョロボン"},
        "姆克鸟": {"english": "Staravia", "japanese": "ムックル"},
        "玛利露": {"english": "Marill", "japanese": "マリル"},
        "毒蔷薇": {"english": "Roselia", "japanese": "ロゼリア"},
        "安瓢虫": {"english": "Ledian", "japanese": "レディアン"},
        "鸭嘴火兽": {"english": "Magmar", "japanese": "ブーバー"},
        "狩猎凤蝶": {"english": "Beautifly", "japanese": "アゲハント"},
        "大王燕": {"english": "Swellow", "japanese": "オオスバメ"},
        "钻天猴": {"english": "Ambipom", "japanese": "エテボース"},
        "伞电蜥": {"english": "Helioptile", "japanese": "エリキテル"},
        "尼多王": {"english": "Nidoking", "japanese": "ニドキング"},
        "七夕青鸟": {"english": "Altaria", "japanese": "チルタリス"},
        "牛蛙君": {"english": "Politoed", "japanese": "ニョロトノ"},
        "巴大蝶": {"english": "Butterfree", "japanese": "バタフリー"},
        "结草贵妇": {"english": "Wormadam", "japanese": "ミノマダム"},
        "碧粉蝶": {"english": "Vivillon", "japanese": "ビビヨン"},
        "波可比": {"english": "Togepi", "japanese": "トゲピー"},
        "月桂叶吧": {"english": "Bayleef", "japanese": "ベイリーフ"},
    },
    
    # Digimon (数码宝贝)
    "数码宝贝": {
        "贝壳兽": {"english": "Shellmon", "japanese": "シェルモン"},
        "恶霸熊仔兽": {"english": "Grizzlymon", "japanese": "グリズモン"},
        "加布兽": {"english": "Gabumon", "japanese": "ガブモン"},
        "基尔兽": {"english": "Guilmon", "japanese": "ギルモン"},
        "V仔兽EX": {"english": "Veedramon EX", "japanese": "ブイドラモンEX"},
    },
    
    # Inuyasha (犬夜叉)
    "犬夜叉": {
        "钢牙": {"english": "Kōga", "japanese": "鋼牙"},
    },
    
    # My Little Pony (小马宝莉)
    "小马宝莉": {
        "余晖烁烁": {"english": "Sunset Shimmer", "japanese": "サンセット・シマー"},
    },
    
    # Other known characters
    "其他": {
        "松坂梅": {"english": "Matsuzaka Ume", "japanese": "松坂梅", "anime": "蜡笔小新"},
        "石井健太郎": {"english": "Ishii Kentarou", "japanese": "石井健太郎", "anime": "蜡笔小新"},
        "露娜玛丽亚·霍克": {"english": "Lunamaria Hawke", "japanese": "ルナマリア・ホーク", "anime": "机动战士高达SEED DESTINY"},
        "珍妮·莉亚莱特": {"english": "Jenny Realight", "japanese": "ジェニー・リアライト", "anime": "机动战士高达SEED"},
        "古妮雅‧哈维": {"english": "Clare", "japanese": "クレア", "anime": "大剑"},
        "艾尔夫曼·斯特劳斯": {"english": "Elfman Strauss", "japanese": "エルフマン・ストラウス", "anime": "妖精的尾巴"},
        "瓦里埃尔": {"english": "Valliere", "japanese": "ヴァリエール", "anime": "零之使魔"},
        "罗姆爷": {"english": "Rom", "japanese": "ロム", "anime": "Re:从零开始的异世界生活"},
        "雪莉·布兰蒂": {"english": "Sherry Blendy", "japanese": "シェリー・ブレンディ", "anime": "机动战士高达00"},
        "小日向奏": {"english": "Kohinata Kanade", "japanese": "小日向奏", "anime": "战姬绝唱"},
        "阿莉耶丝": {"english": "Aries", "japanese": "アリエス", "anime": "FAIRY TAIL"},
        "罗玛丽·斯通": {"english": "Romary Stone", "japanese": "ロマリー・ストーン", "anime": "机动战士高达AGE"},
        "雨宫勇气": {"english": "Amamiya Yuki", "japanese": "雨宮勇気", "anime": "境界触发者"},
        "花鹿·路易莎·陆深·伯斯华斯": {"english": "Kajika Louisa Vossen", "japanese": "花鹿・ルイーザ・陸深・フォッセンワース", "anime": "爱丽丝学园"},
        "修帝": {"english": "Shooty", "japanese": "シューティー", "anime": "精灵宝可梦"},
        "菩提大伯": {"english": "Bodhi", "japanese": "ボーディ", "anime": "其他"},
        "久住舞子": {"english": "Hisazumi Maiko", "japanese": "久住舞子", "anime": "名侦探柯南"},
        "疏楼龙宿": {"english": "Shulou Longxiu", "japanese": "疏楼龍宿", "anime": "霹雳布袋戏"},
        "本多·正纯": {"english": "Honda Masazumi", "japanese": "本多・正純", "anime": "境界线上的地平线"},
        "不破冰菓": {"english": "Fuwa Hyōka", "japanese": "不破氷菓", "anime": "我的青春恋爱物语果然有问题"},
        "不破春斗": {"english": "Fuwa Haruto", "japanese": "不破春斗", "anime": "如果有妹妹就好了"},
        "本田珠辉": {"english": "Honda Tamaki", "japanese": "本田たまき", "anime": "NEW GAME!"},
        "本田透": {"english": "Honda Tōru", "japanese": "本田透", "anime": "水果篮子"},
        "木曾": {"english": "Kiso", "japanese": "木曾", "anime": "其他"},
        "赵琳": {"english": "Zhao Lin", "japanese": "趙琳", "anime": "熊出没"},
        "蟹老板": {"english": "Mr. Krabs", "japanese": "クラブス", "anime": "海绵宝宝"},
        "猪妈妈": {"english": "Mummy Pig", "japanese": "マミーピッグ", "anime": "小猪佩奇"},
    }
}

def parse_file(file_path: str) -> List[Dict]:
    """Parse the file and extract all entries."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
                entries.append(entry)
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    return entries

def find_character_info(chinese_name: str) -> Dict[str, Optional[str]]:
    """Find English and Japanese names for a character."""
    result = {
        'english_name': None,
        'japanese_name': None,
        'anime_series': None
    }
    
    # Search through all anime series
    for series_name, characters in ANIME_SERIES_MAPPINGS.items():
        if chinese_name in characters:
            char_info = characters[chinese_name]
            result['english_name'] = char_info.get('english')
            result['japanese_name'] = char_info.get('japanese')
            result['anime_series'] = char_info.get('anime', series_name)
            break
    
    return result

def main():
    input_file = "data/icartoonface_rectest_idInfo.txt"
    output_file = "data/chinese_to_english_names.json"
    
    print(f"Reading file: {input_file}")
    entries = parse_file(input_file)
    print(f"Found {len(entries)} entries")
    
    # Group by character name
    character_dict = {}
    for entry in entries:
        name = entry.get('name', '')
        if name not in character_dict:
            character_dict[name] = {
                'chinese_name': name,
                'urls': [],
                'ids': []
            }
        if entry.get('url'):
            character_dict[name]['urls'].append(entry.get('url'))
        character_dict[name]['ids'].append(entry.get('id'))
    
    print(f"Unique characters: {len(character_dict)}")
    
    # Find English/Japanese names
    results = []
    found_count = 0
    
    for name, info in character_dict.items():
        char_info = find_character_info(name)
        
        result = {
            'chinese_name': name,
            'urls': info['urls'],
            'ids': info['ids'],
            'english_name': char_info['english_name'],
            'japanese_name': char_info['japanese_name'],
            'anime_series': char_info['anime_series']
        }
        
        if char_info['english_name']:
            found_count += 1
        
        results.append(result)
    
    # Save results
    output_data = {
        'total_entries': len(entries),
        'unique_characters': len(character_dict),
        'characters_with_names': found_count,
        'characters': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_file}")
    print(f"Found English/Japanese names for {found_count} characters")
    print(f"Coverage: {found_count/len(character_dict)*100:.1f}%")

if __name__ == "__main__":
    main()






