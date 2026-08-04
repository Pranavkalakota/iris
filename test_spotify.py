import iris_spotify as sp

ok, result = sp.resolve_song("Kendrick Lamar")
print("resolve_song:", ok, result)

if ok:
    print("\nPlaylists:")
    for p in sp.list_playlists():
        print(" -", p["name"])