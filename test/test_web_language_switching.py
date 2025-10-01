"""
Web UI言語切り替え機能のテストスクリプト
"""
import requests
import json

def test_language_api():
    """言語切り替えAPIの動作確認"""
    base_url = "http://127.0.0.1:5000"

    print("=== Web UI言語切り替え機能テスト ===\n")

    # セッションを使用して状態を保持
    session = requests.Session()

    print("1. 現在の言語設定を取得...")
    try:
        response = session.get(f"{base_url}/api/get-language")
        data = response.json()
        print(f"   現在の言語: {data.get('language', 'N/A')}")
        print(f"   利用可能な言語: {data.get('available_languages', [])}")
    except Exception as e:
        print(f"   エラー: {e}")

    print("\n2. 日本語に切り替えテスト...")
    try:
        response = session.post(f"{base_url}/api/set-language",
                              headers={'Content-Type': 'application/json'},
                              data=json.dumps({'language': 'ja'}))
        data = response.json()
        if data.get('success'):
            print(f"   [成功] {data.get('message')}")
        else:
            print(f"   [失敗] {data.get('error')}")
    except Exception as e:
        print(f"   エラー: {e}")

    print("\n3. 英語に切り替えテスト...")
    try:
        response = session.post(f"{base_url}/api/set-language",
                              headers={'Content-Type': 'application/json'},
                              data=json.dumps({'language': 'en'}))
        data = response.json()
        if data.get('success'):
            print(f"   [成功] {data.get('message')}")
        else:
            print(f"   [失敗] {data.get('error')}")
    except Exception as e:
        print(f"   エラー: {e}")

    print("\n4. 無効な言語設定テスト...")
    try:
        response = session.post(f"{base_url}/api/set-language",
                              headers={'Content-Type': 'application/json'},
                              data=json.dumps({'language': 'fr'}))
        data = response.json()
        if not data.get('success'):
            print(f"   [正常] エラーが正しく検出されました: {data.get('error')}")
        else:
            print(f"   [警告] 無効な言語が受け入れられました")
    except Exception as e:
        print(f"   エラー: {e}")

    print("\n5. 最終確認：現在の言語設定...")
    try:
        response = session.get(f"{base_url}/api/get-language")
        data = response.json()
        print(f"   最終的な言語設定: {data.get('language', 'N/A')}")
    except Exception as e:
        print(f"   エラー: {e}")

    print("\nテスト完了")

if __name__ == "__main__":
    print("Webサーバーが http://127.0.0.1:5000 で実行されていることを確認してください")
    print("サーバーを起動するには: python web_interface.py")
    print("\n注意: demo_new.htmlテンプレートに言語切り替えボタンが追加されました")
    print("ブラウザで http://127.0.0.1:5000 にアクセスして右上の言語切り替えボタンを確認してください")
    print("\nAPIテストを開始するには Enter を押してください...")
    input()

    test_language_api()