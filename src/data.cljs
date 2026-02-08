(ns data)

(def by-key-map
  {
"APL@/ (reduce)@-@1" {:lang "APL" :algo "/ (reduce)" :lib "-" :id 1 :doc "http://microapl.com/apl_help/ch_020_020_800.htm" :sig "-" :expr false}
"APL@⌿ (reduce first)@-@1" {:lang "APL" :algo "⌿ (reduce first)" :lib "-" :id 1 :doc "https://microapl.com/apl_help/ch_020_020_810.htm" :sig "-" :expr false}
"Agda@foldl@various@1" {:lang "Agda" :algo "foldl" :lib "various" :id 1 :doc "https://agda.github.io/agda-stdlib/v2.2/Effect.Foldable.html#1843" :sig "-" :expr false}
"BQN@˝ (insert)@-@1" {:lang "BQN" :algo "˝ (insert)" :lib "-" :id 1 :doc "https://mlochbaum.github.io/BQN/help/insert.html" :sig "-" :expr false}
"BQN@´ (fold)@-@1" {:lang "BQN" :algo "´ (fold)" :lib "-" :id 1 :doc "https://mlochbaum.github.io/BQN/help/fold.html" :sig "-" :expr false}
"C#@Aggregate@Enumerable@1" {:lang "C#" :algo "Aggregate" :lib "Enumerable" :id 1 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.aggregate" :sig "-" :expr false}
"C++@accumulate@<numeric>@1" {:lang "C++" :algo "accumulate" :lib "<numeric>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/accumulate" :sig "-" :expr false}
"C++@fold_left@<algorithm>@1" {:lang "C++" :algo "fold_left" :lib "<algorithm>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/fold_left" :sig "-" :expr false}
"C++@fold_left_first@<algorithm>@1" {:lang "C++" :algo "fold_left_first" :lib "<algorithm>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/fold_left_first" :sig "-" :expr false}
"C++@fold_right@<algorithm>@1" {:lang "C++" :algo "fold_right" :lib "<algorithm>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/fold_right" :sig "-" :expr false}
"C++@fold_right_last@<algorithm>@1" {:lang "C++" :algo "fold_right_last" :lib "<algorithm>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/fold_right_last" :sig "-" :expr false}
"C++@reduce@<numeric>@1" {:lang "C++" :algo "reduce" :lib "<numeric>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/reduce" :sig "-" :expr false}
"CUDA@reduce@Thrust@1" {:lang "CUDA" :algo "reduce" :lib "Thrust" :id 1 :doc "https://thrust.github.io/doc/group__reductions_ga43eea9a000f912716189687306884fc7.html" :sig "-" :expr false}
"Clojure@reduce@core@1" {:lang "Clojure" :algo "reduce" :lib "core" :id 1 :doc "https://clojuredocs.org/clojure.core/reduce" :sig "-" :expr false}
"Common Lisp@reduce@-@1" {:lang "Common Lisp" :algo "reduce" :lib "-" :id 1 :doc "https://novaspec.org/cl/f_reduce" :sig "-" :expr false}
"D@reduce@algorithm.iteration@1" {:lang "D" :algo "reduce" :lib "algorithm.iteration" :id 1 :doc "https://dlang.org/library/std/algorithm/iteration/reduce.html" :sig "-" :expr false}
"D@fold@algorithm.iteration@1" {:lang "D" :algo "fold" :lib "algorithm.iteration" :id 1 :doc "https://dlang.org/library/std/algorithm/iteration/fold.html" :sig "-" :expr false}
"Elixir@reduce@Enum@1" {:lang "Elixir" :algo "reduce" :lib "Enum" :id 1 :doc "https://hexdocs.pm/elixir/Enum.html#reduce/2" :sig "-" :expr false}
"Erlang@foldl@lists@31" {:lang "Erlang" :algo "foldl" :lib "lists" :id 31 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#foldl/3" :sig "-" :expr false}
"Excel@reduce@-@1" {:lang "Excel" :algo "reduce" :lib "-" :id 1 :doc "https://support.microsoft.com/en-us/office/reduce-function-42e39910-b345-45f3-84b8-0642b568b7cb" :sig "-" :expr false}
"Haskell@foldl@Data.List@1" {:lang "Haskell" :algo "foldl" :lib "Data.List" :id 1 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:foldl" :sig "-" :expr false}
"J@/ (insert)@-@1" {:lang "J" :algo "/ (insert)" :lib "-" :id 1 :doc "https://code.jsoftware.com/wiki/Vocabulary/slash" :sig "-" :expr false}
"Java@reduce@Streams@1" {:lang "Java" :algo "reduce" :lib "Streams" :id 1 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#reduce-java.util.function.BinaryOperator-" :sig "-" :expr false}
"JavaScript@reduce@-@1" {:lang "JavaScript" :algo "reduce" :lib "-" :id 1 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reduce" :sig "-" :expr false}
"JavaScript@reduce@ramda@1" {:lang "JavaScript" :algo "reduce" :lib "ramda" :id 1 :doc "https://ramdajs.com/docs/#reduce" :sig "-" :expr false}
"Julia@reduce@Base@1" {:lang "Julia" :algo "reduce" :lib "Base" :id 1 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.reduce-Tuple%7BAny,%20Any%7D" :sig "-" :expr false}
"Kap@/ (reduce)@-@1" {:lang "Kap" :algo "/ (reduce)" :lib "-" :id 1 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_reduce" :sig "-" :expr false}
"Kotlin@reduce@collections@1" {:lang "Kotlin" :algo "reduce" :lib "collections" :id 1 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/reduce.html" :sig "-" :expr false}
"Kotlin@fold@collections@1" {:lang "Kotlin" :algo "fold" :lib "collections" :id 1 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/fold.html" :sig "-" :expr false}
"Mathematica@Fold@-@1" {:lang "Mathematica" :algo "Fold" :lib "-" :id 1 :doc "https://reference.wolfram.com/language/ref/Fold.html" :sig "-" :expr false}
"Pharo@inject:into:@Collection@1" {:lang "Pharo" :algo "inject:into:" :lib "Collection" :id 1 :doc "" :sig "-" :expr false}
"Pharo@reduce:@Collection@1" {:lang "Pharo" :algo "reduce:" :lib "Collection" :id 1 :doc "" :sig "-" :expr false}
"Pharo@fold:@Collection@1" {:lang "Pharo" :algo "fold:" :lib "Collection" :id 1 :doc "" :sig "-" :expr false}
"Python@reduce@functools@1" {:lang "Python" :algo "reduce" :lib "functools" :id 1 :doc "https://docs.python.org/3/library/functools.html#functools.reduce" :sig "-" :expr false}
"Racket@foldl@base@1" {:lang "Racket" :algo "foldl" :lib "base" :id 1 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Fprivate%2Flist..rkt%29._foldl%29%29" :sig "-" :expr false}
"Ruby@reduce@Enumerable@1" {:lang "Ruby" :algo "reduce" :lib "Enumerable" :id 1 :doc "https://apidock.com/ruby/Enumerable/reduce" :sig "-" :expr false}
"Ruby@inject@Enumerable@1" {:lang "Ruby" :algo "inject" :lib "Enumerable" :id 1 :doc "https://apidock.com/ruby/Enumerable/inject" :sig "-" :expr false}
"Rust@fold@iter@1" {:lang "Rust" :algo "fold" :lib "iter" :id 1 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.fold" :sig "-" :expr false}
"Scala@fold@various@1" {:lang "Scala" :algo "fold" :lib "various" :id 1 :doc "https://scala-lang.org/api/current/scala/collection/IterableOnceOps.html#fold-fffff9e1" :sig "-" :expr false}
"Scala@reduce@various@1" {:lang "Scala" :algo "reduce" :lib "various" :id 1 :doc "https://scala-lang.org/api/current/scala/collection/IterableOnceOps.html#reduce-fffff3d4" :sig "-" :expr false}
"TinyAPL@/ (reduce)@-@1" {:lang "TinyAPL" :algo "/ (reduce)" :lib "-" :id 1 :doc "https://tinyapl.rubenverg.com/docs/primitive/reduce" :sig "-" :expr false}
"Uiua@∧ (fold)@-@1" {:lang "Uiua" :algo "∧ (fold)" :lib "-" :id 1 :doc "https://www.uiua.org/docs/fold" :sig "-" :expr false}
"Uiua@/ (reduce)@-@1" {:lang "Uiua" :algo "/ (reduce)" :lib "-" :id 1 :doc "https://www.uiua.org/docs/reduce" :sig "-" :expr false}
"q@over@-@1" {:lang "q" :algo "over" :lib "-" :id 1 :doc "https://code.kx.com/q/ref/over/" :sig "-" :expr false}
"☾@ſ  (fold)@-@1" {:lang "☾" :algo "ſ  (fold)" :lib "-" :id 1 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@¨ (each)@-@2" {:lang "APL" :algo "¨ (each)" :lib "-" :id 2 :doc "http://microapl.com/apl_help/ch_020_020_900.htm" :sig "-" :expr false}
"Agda@map@various@2" {:lang "Agda" :algo "map" :lib "various" :id 2 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#1634" :sig "-" :expr false}
"BQN@¨ (each)@-@2" {:lang "BQN" :algo "¨ (each)" :lib "-" :id 2 :doc "https://mlochbaum.github.io/BQN/doc/map.html" :sig "-" :expr false}
"BQN@˘ (cells)@-@2" {:lang "BQN" :algo "˘ (cells)" :lib "-" :id 2 :doc "https://mlochbaum.github.io/BQN/doc/rank.html" :sig "-" :expr false}
"C++@transform@<algorithm>@2" {:lang "C++" :algo "transform" :lib "<algorithm>" :id 2 :doc "https://en.cppreference.com/w/cpp/algorithm/transform" :sig "-" :expr false}
"Clojure@map@core@2" {:lang "Clojure" :algo "map" :lib "core" :id 2 :doc "https://clojuredocs.org/clojure.core/map" :sig "-" :expr false}
"Common Lisp@map@-@2" {:lang "Common Lisp" :algo "map" :lib "-" :id 2 :doc "https://novaspec.org/cl/f_map" :sig "-" :expr false}
"D@map@algorithm.iteration@2" {:lang "D" :algo "map" :lib "algorithm.iteration" :id 2 :doc "https://dlang.org/library/std/algorithm/iteration/map.html" :sig "-" :expr false}
"Elixir@map@Enum@2" {:lang "Elixir" :algo "map" :lib "Enum" :id 2 :doc "https://hexdocs.pm/elixir/Enum.html#map/2" :sig "-" :expr false}
"Erlang@map@lists@31" {:lang "Erlang" :algo "map" :lib "lists" :id 31 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#map/2" :sig "-" :expr false}
"Excel@map@-@2" {:lang "Excel" :algo "map" :lib "-" :id 2 :doc "https://support.microsoft.com/en-us/office/map-function-48006093-f97c-47c1-bfcc-749263bb1f01" :sig "-" :expr false}
"Gleam@map@list@2" {:lang "Gleam" :algo "map" :lib "list" :id 2 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#map" :sig "-" :expr false}
"Haskell@map@Data.List@2" {:lang "Haskell" :algo "map" :lib "Data.List" :id 2 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:map" :sig "-" :expr false}
"J@each@-@2" {:lang "J" :algo "each" :lib "-" :id 2 :doc "https://www.jsoftware.com/help/primer/each.htm" :sig "-" :expr false}
"Java@map@Streams@2" {:lang "Java" :algo "map" :lib "Streams" :id 2 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#map-java.util.function.Function-" :sig "-" :expr false}
"JavaScript@map@-@2" {:lang "JavaScript" :algo "map" :lib "-" :id 2 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map" :sig "-" :expr false}
"JavaScript@map@ramda@2" {:lang "JavaScript" :algo "map" :lib "ramda" :id 2 :doc "https://ramdajs.com/docs/#map" :sig "-" :expr false}
"Julia@map@Base@2" {:lang "Julia" :algo "map" :lib "Base" :id 2 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.map" :sig "-" :expr false}
"Kotlin@map@collections@2" {:lang "Kotlin" :algo "map" :lib "collections" :id 2 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/map.html" :sig "-" :expr false}
"Mathematica@/@ (Map)@-@2" {:lang "Mathematica" :algo "/@ (Map)" :lib "-" :id 2 :doc "http://reference.wolfram.com/language/ref/Map.html" :sig "-" :expr false}
"Python@map@-@2" {:lang "Python" :algo "map" :lib "-" :id 2 :doc "https://docs.python.org/3/library/functions.html#map" :sig "-" :expr false}
"Racket@map@base@2" {:lang "Racket" :algo "map" :lib "base" :id 2 :doc "https://docs.racket-lang.org/reference/pairs.html?q=map#%28def._%28%28lib._racket%2Fprivate%2Fmap..rkt%29._map%29%29" :sig "-" :expr false}
"Ruby@map@Enumerable@2" {:lang "Ruby" :algo "map" :lib "Enumerable" :id 2 :doc "https://apidock.com/ruby/Enumerable/map" :sig "-" :expr false}
"Rust@map@iter@2" {:lang "Rust" :algo "map" :lib "iter" :id 2 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.map" :sig "-" :expr false}
"Scala@map@various@2" {:lang "Scala" :algo "map" :lib "various" :id 2 :doc "https://scala-lang.org/api/current/scala/collection/IterableOps.html#map-5d3" :sig "-" :expr false}
"TinyAPL@¨ (each)@-@2" {:lang "TinyAPL" :algo "¨ (each)" :lib "-" :id 2 :doc "https://tinyapl.rubenverg.com/docs/primitive/each" :sig "-" :expr false}
"Uiua@∵ (each)@-@2" {:lang "Uiua" :algo "∵ (each)" :lib "-" :id 2 :doc "https://www.uiua.org/docs/each" :sig "-" :expr false}
"Uiua@≡ (rows)@-@2" {:lang "Uiua" :algo "≡ (rows)" :lib "-" :id 2 :doc "https://www.uiua.org/docs/rows" :sig "-" :expr false}
"q@each@-@2" {:lang "q" :algo "each" :lib "-" :id 2 :doc "https://code.kx.com/q/ref/each/" :sig "-" :expr false}
"☾@ᴍ  (map)@-@2" {:lang "☾" :algo "ᴍ  (map)" :lib "-" :id 2 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@/ (compress)@-@3" {:lang "APL" :algo "/ (compress)" :lib "-" :id 3 :doc "http://microapl.com/apl_help/ch_020_020_840.htm" :sig "-" :expr false}
"Agda@filter@various@3" {:lang "Agda" :algo "filter" :lib "various" :id 3 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#10687" :sig "-" :expr false}
"C#@Where@Enumerable@3" {:lang "C#" :algo "Where" :lib "Enumerable" :id 3 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.where?view=netcore-3.1" :sig "-" :expr false}
"C++@filter@range-v3@3" {:lang "C++" :algo "filter" :lib "range-v3" :id 3 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-" :expr false}
"C++@filter@<ranges>@3" {:lang "C++" :algo "filter" :lib "<ranges>" :id 3 :doc "https://en.cppreference.com/w/cpp/ranges/filter_view" :sig "-" :expr false}
"Clojure@filter@core@3" {:lang "Clojure" :algo "filter" :lib "core" :id 3 :doc "https://clojuredocs.org/clojure.core/filter" :sig "-" :expr false}
"D@filter@algorithm@3" {:lang "D" :algo "filter" :lib "algorithm" :id 3 :doc "https://dlang.org/library/std/algorithm/iteration/filter.html" :sig "-" :expr false}
"Elixir@filter@Enum@3" {:lang "Elixir" :algo "filter" :lib "Enum" :id 3 :doc "https://hexdocs.pm/elixir/Enum.html#filter/2" :sig "-" :expr false}
"Erlang@filter@lists@21" {:lang "Erlang" :algo "filter" :lib "lists" :id 21 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#filter/2" :sig "-" :expr false}
"Excel@filter@-@3" {:lang "Excel" :algo "filter" :lib "-" :id 3 :doc "https://support.microsoft.com/en-us/office/filter-function-f4f7cb66-82eb-4767-8f7c-4877ad80c759" :sig "-" :expr false}
"F#@where@various@3" {:lang "F#" :algo "where" :lib "various" :id 3 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-arraymodule.html#where" :sig "-" :expr false}
"F#@filter@List@3" {:lang "F#" :algo "filter" :lib "List" :id 3 :doc "https://msdn.microsoft.com/visualfsharpdocs/conceptual/list.filter%5b%27t%5d-function-%5bfsharp%5d" :sig "-" :expr false}
"Haskell@filter@Data.List@3" {:lang "Haskell" :algo "filter" :lib "Data.List" :id 3 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:filter" :sig "-" :expr false}
"Java@filter@Streams@3" {:lang "Java" :algo "filter" :lib "Streams" :id 3 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#filter-java.util.function.Predicate-" :sig "-" :expr false}
"JavaScript@filter@-@3" {:lang "JavaScript" :algo "filter" :lib "-" :id 3 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter" :sig "-" :expr false}
"JavaScript@filter@ramda@3" {:lang "JavaScript" :algo "filter" :lib "ramda" :id 3 :doc "https://ramdajs.com/docs/#filter" :sig "-" :expr false}
"Julia@filter@Base@3" {:lang "Julia" :algo "filter" :lib "Base" :id 3 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.filter" :sig "-" :expr false}
"Kotlin@filter@collections@3" {:lang "Kotlin" :algo "filter" :lib "collections" :id 3 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/filter.html" :sig "-" :expr false}
"Mathematica@Select@-@3" {:lang "Mathematica" :algo "Select" :lib "-" :id 3 :doc "http://reference.wolfram.com/language/ref/Select.html" :sig "-" :expr false}
"OCaml@filter@List@3" {:lang "OCaml" :algo "filter" :lib "List" :id 3 :doc "https://ocaml.org/releases/4.10/htmlman/libref/List.html" :sig "-" :expr false}
"Python@filter@-@3" {:lang "Python" :algo "filter" :lib "-" :id 3 :doc "https://docs.python.org/3/library/functions.html#filter" :sig "-" :expr false}
"Racket@filter@base@3" {:lang "Racket" :algo "filter" :lib "base" :id 3 :doc "https://docs.racket-lang.org/reference/pairs.html?q=filter#%28def._%28%28lib._racket%2Fprivate%2Flist..rkt%29._filter%29%29" :sig "-" :expr false}
"Ruby@filter@Array@3" {:lang "Ruby" :algo "filter" :lib "Array" :id 3 :doc "https://apidock.com/ruby/Array/filter" :sig "-" :expr false}
"Rust@filter@iter@3" {:lang "Rust" :algo "filter" :lib "iter" :id 3 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.filter" :sig "-" :expr false}
"Scala@filter@various@3" {:lang "Scala" :algo "filter" :lib "various" :id 3 :doc "https://scala-lang.org/api/current/scala/collection/ArrayOps.html#filter-c24" :sig "-" :expr false}
"☾@󰈲  (filter positive)@-@3" {:lang "☾" :algo "󰈲  (filter positive)" :lib "-" :id 3 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@\\ (scan)@-@4" {:lang "APL" :algo "\\ (scan)" :lib "-" :id 4 :doc "http://microapl.com/apl_help/ch_020_020_820.htm" :sig "-" :expr false}
"Agda@scanl@various@4" {:lang "Agda" :algo "scanl" :lib "various" :id 4 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Scans.Base.html#1105" :sig "-" :expr false}
"BQN@ ` @-@4" {:lang "BQN" :algo " ` " :lib "-" :id 4 :doc "https://mlochbaum.github.io/BQN/help/scan.html" :sig "-" :expr false}
"C++@inclusive_scan@<numeric>@4" {:lang "C++" :algo "inclusive_scan" :lib "<numeric>" :id 4 :doc "https://en.cppreference.com/w/cpp/algorithm/inclusive_scan" :sig "-" :expr false}
"C++@partial_sum@<numeric>@4" {:lang "C++" :algo "partial_sum" :lib "<numeric>" :id 4 :doc "https://en.cppreference.com/w/cpp/algorithm/partial_sum" :sig "-" :expr false}
"CUDA@inclusive_scan@Thrust@4" {:lang "CUDA" :algo "inclusive_scan" :lib "Thrust" :id 4 :doc "https://thrust.github.io/doc/group__prefixsums_gafb24ad76101263038b0acaddc094d70a.html" :sig "-" :expr false}
"Clojure@reductions@core@4" {:lang "Clojure" :algo "reductions" :lib "core" :id 4 :doc "https://clojuredocs.org/clojure.core/reductions" :sig "-" :expr false}
"D@cumulativeFold@algorithm.iteration@4" {:lang "D" :algo "cumulativeFold" :lib "algorithm.iteration" :id 4 :doc "https://dlang.org/library/std/algorithm/iteration/cumulative_fold.html" :sig "-" :expr false}
"Elixir@scan@Enum@4" {:lang "Elixir" :algo "scan" :lib "Enum" :id 4 :doc "https://hexdocs.pm/elixir/Enum.html#scan/2" :sig "-" :expr false}
"Excel@scan@-@4" {:lang "Excel" :algo "scan" :lib "-" :id 4 :doc "https://support.microsoft.com/en-us/office/scan-function-d58dfd11-9969-4439-b2dc-e7062724de29" :sig "-" :expr false}
"F#@scan@Seq@4" {:lang "F#" :algo "scan" :lib "Seq" :id 4 :doc "https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/sequences" :sig "-" :expr false}
"Gleam@scan@list@4" {:lang "Gleam" :algo "scan" :lib "list" :id 4 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#scan" :sig "-" :expr false}
"Haskell@scanl1@Data.List@4" {:lang "Haskell" :algo "scanl1" :lib "Data.List" :id 4 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:scanl1" :sig "-" :expr false}
"J@\\ (prefix)@-@4" {:lang "J" :algo "\\ (prefix)" :lib "-" :id 4 :doc "https://code.jsoftware.com/wiki/Vocabulary/bslash" :sig "-" :expr false}
"JavaScript@scan@ramda@4" {:lang "JavaScript" :algo "scan" :lib "ramda" :id 4 :doc "https://ramdajs.com/docs/#scan" :sig "-" :expr false}
"Kap@\\ (scan)@-@4" {:lang "Kap" :algo "\\ (scan)" :lib "-" :id 4 :doc "https://kapdemo.dhsdevelopments.com/reference.html#scan" :sig "-" :expr false}
"Kotlin@runningReduce@collections@4" {:lang "Kotlin" :algo "runningReduce" :lib "collections" :id 4 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/running-reduce.html" :sig "-" :expr false}
"Kotlin@runningFold@collections@4" {:lang "Kotlin" :algo "runningFold" :lib "collections" :id 4 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/running-fold.html" :sig "-" :expr false}
"Mathematica@FoldList@-@4" {:lang "Mathematica" :algo "FoldList" :lib "-" :id 4 :doc "http://reference.wolfram.com/language/ref/FoldList.html" :sig "-" :expr false}
"Python@accumulate@itertools@4" {:lang "Python" :algo "accumulate" :lib "itertools" :id 4 :doc "https://docs.python.org/3/library/itertools.html#itertools.accumulate" :sig "-" :expr false}
"Python@cumsum@RAPIDS cuDF@4" {:lang "Python" :algo "cumsum" :lib "RAPIDS cuDF" :id 4 :doc "https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.core.series.Series.cumsum" :sig "-" :expr false}
"Python@cumsum@pandas@4" {:lang "Python" :algo "cumsum" :lib "pandas" :id 4 :doc "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cumsum.html?highlight=cumsum#pandas.Series.cumsum" :sig "-" :expr false}
"Rust@scan@iter@4" {:lang "Rust" :algo "scan" :lib "iter" :id 4 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.scan" :sig "-" :expr false}
"Scala@scan@various@4" {:lang "Scala" :algo "scan" :lib "various" :id 4 :doc "https://scala-lang.org/api/current/scala/collection/IterableOps.html#scan-fffff9e1" :sig "-" :expr false}
"Swift@reductions@Algorithms@4" {:lang "Swift" :algo "reductions" :lib "Algorithms" :id 4 :doc "https://swiftpackageindex.com/apple/swift-algorithms/main/documentation/algorithms/swift/sequence/reductions(_:)" :sig "-" :expr false}
"TinyAPL@↟ (on prefixes)@-@4" {:lang "TinyAPL" :algo "↟ (on prefixes)" :lib "-" :id 4 :doc "https://tinyapl.rubenverg.com/docs/primitive/on_prefixes" :sig "-" :expr false}
"Uiua@\\ (scan)@-@4" {:lang "Uiua" :algo "\\ (scan)" :lib "-" :id 4 :doc "https://www.uiua.org/docs/scan" :sig "-" :expr false}
"q@scan@-@4" {:lang "q" :algo "scan" :lib "-" :id 4 :doc "https://code.kx.com/q/ref/over/" :sig "-" :expr false}
"☾@Ϝ  (accumulate)@-@4" {:lang "☾" :algo "Ϝ  (accumulate)" :lib "-" :id 4 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@/ (n-wise reduce)@-@5" {:lang "APL" :algo "/ (n-wise reduce)" :lib "-" :id 5 :doc "http://microapl.com/apl_help/ch_020_020_800.htm" :sig "-" :expr false}
"C++@adjacent_difference@<numeric>@5" {:lang "C++" :algo "adjacent_difference" :lib "<numeric>" :id 5 :doc "https://en.cppreference.com/w/cpp/algorithm/adjacent_difference" :sig "-" :expr false}
"C++@adjacent_transform@<ranges>@5" {:lang "C++" :algo "adjacent_transform" :lib "<ranges>" :id 5 :doc "https://en.cppreference.com/w/cpp/ranges/adjacent_transform_view" :sig "-" :expr false}
"CUDA@adjacent_difference@Thrust@5" {:lang "CUDA" :algo "adjacent_difference" :lib "Thrust" :id 5 :doc "https://thrust.github.io/doc/group__transformations_gaa41d309b53fa03bf13fe35a184148400.html" :sig "-" :expr false}
"Haskell@mapAdjacent@Data.List.HT@5" {:lang "Haskell" :algo "mapAdjacent" :lib "Data.List.HT" :id 5 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:mapAdjacent" :sig "-" :expr false}
"Kotlin@zipWithNext@collections@5" {:lang "Kotlin" :algo "zipWithNext" :lib "collections" :id 5 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/zip-with-next.html" :sig "-" :expr false}
"Python@pairwise@itertools@5" {:lang "Python" :algo "pairwise" :lib "itertools" :id 5 :doc "https://docs.python.org/3/library/itertools.html#itertools.pairwise" :sig "-" :expr false}
"TinyAPL@↡ (on infixes)@-@5" {:lang "TinyAPL" :algo "↡ (on infixes)" :lib "-" :id 5 :doc "https://tinyapl.rubenverg.com/docs/primitive/on_infixes" :sig "-" :expr false}
"Uiua@⧈ (stencil)@-@5" {:lang "Uiua" :algo "⧈ (stencil)" :lib "-" :id 5 :doc "https://www.uiua.org/docs/stencil" :sig "-" :expr false}
"q@prior@-@5" {:lang "q" :algo "prior" :lib "-" :id 5 :doc "https://code.kx.com/q/ref/prior/" :sig "-" :expr false}
"APL@∘. (outer product)@-@6" {:lang "APL" :algo "∘. (outer product)" :lib "-" :id 6 :doc "http://microapl.com/apl_help/ch_020_020_890.htm" :sig "-" :expr false}
"BQN@⌜ (table)@-@6" {:lang "BQN" :algo "⌜ (table)" :lib "-" :id 6 :doc "https://mlochbaum.github.io/BQN/help/table.html" :sig "-" :expr false}
"Clojure@outer-product@core.matrix@6" {:lang "Clojure" :algo "outer-product" :lib "core.matrix" :id 6 :doc "https://mikera.github.io/core.matrix/doc/clojure.core.matrix.html#var-outer-product" :sig "-" :expr false}
"Haskell@outerProduct@Data.List.HT@6" {:lang "Haskell" :algo "outerProduct" :lib "Data.List.HT" :id 6 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:outerProduct" :sig "-" :expr false}
"J@/ (table)@-@6" {:lang "J" :algo "/ (table)" :lib "-" :id 6 :doc "https://code.jsoftware.com/wiki/Vocabulary/slash#dyadic" :sig "-" :expr false}
"Kap@⌻ (outer product)@-@6" {:lang "Kap" :algo "⌻ (outer product)" :lib "-" :id 6 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_outer_product" :sig "-" :expr false}
"Mathematica@Outer@-@6" {:lang "Mathematica" :algo "Outer" :lib "-" :id 6 :doc "http://reference.wolfram.com/language/ref/Outer.html" :sig "-" :expr false}
"Python@outer@NumPy@6" {:lang "Python" :algo "outer" :lib "NumPy" :id 6 :doc "https://numpy.org/doc/stable/reference/generated/numpy.outer.html" :sig "-" :expr false}
"R@outer@-@6" {:lang "R" :algo "outer" :lib "-" :id 6 :doc "https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/outer" :sig "-" :expr false}
"TinyAPL@⊞ (table)@-@6" {:lang "TinyAPL" :algo "⊞ (table)" :lib "-" :id 6 :doc "https://tinyapl.rubenverg.com/docs/primitive/table" :sig "-" :expr false}
"Uiua@⊞ (table)@-@6" {:lang "Uiua" :algo "⊞ (table)" :lib "-" :id 6 :doc "https://www.uiua.org/docs/table" :sig "-" :expr false}
"☾@⊞  (table product)@-@6" {:lang "☾" :algo "⊞  (table product)" :lib "-" :id 6 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@. (inner product)@-@7" {:lang "APL" :algo ". (inner product)" :lib "-" :id 7 :doc "http://microapl.com/apl_help/ch_020_020_880.htm" :sig "-" :expr false}
"C++@inner_product@<numeric>@7" {:lang "C++" :algo "inner_product" :lib "<numeric>" :id 7 :doc "https://en.cppreference.com/w/cpp/algorithm/inner_product" :sig "-" :expr false}
"D@dotProduct@std.numeric@7" {:lang "D" :algo "dotProduct" :lib "std.numeric" :id 7 :doc "https://dlang.org/library/std/numeric/dot_product.html" :sig "-" :expr false}
"J@. (Matrix Product)@-@7" {:lang "J" :algo ". (Matrix Product)" :lib "-" :id 7 :doc "https://code.jsoftware.com/wiki/Vocabulary/dot#dyadic" :sig "-" :expr false}
"Mathematica@Inner@-@7" {:lang "Mathematica" :algo "Inner" :lib "-" :id 7 :doc "http://reference.wolfram.com/language/ref/Inner.html" :sig "-" :expr false}
"TinyAPL@∙ (inner product)@-@7" {:lang "TinyAPL" :algo "∙ (inner product)" :lib "-" :id 7 :doc "https://tinyapl.rubenverg.com/docs/primitive/inner_product" :sig "-" :expr false}
"APL@⍳ (index generator)@-@8" {:lang "APL" :algo "⍳ (index generator)" :lib "-" :id 8 :doc "http://microapl.com/apl_help/ch_020_020_150.htm" :sig "-" :expr false}
"Agda@upTo@Data.List@8" {:lang "Agda" :algo "upTo" :lib "Data.List" :id 8 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#6689" :sig "-" :expr false}
"Agda@allFin@Data.Vec@8" {:lang "Agda" :algo "allFin" :lib "Data.Vec" :id 8 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Vec.Base.html#6708" :sig "-" :expr false}
"C++@iota@<numeric>@8" {:lang "C++" :algo "iota" :lib "<numeric>" :id 8 :doc "https://en.cppreference.com/w/cpp/algorithm/iota" :sig "-" :expr false}
"CUDA@sequence@Thrust@8" {:lang "CUDA" :algo "sequence" :lib "Thrust" :id 8 :doc "https://thrust.github.io/doc/group__transformations_ga233a3db0c5031023c8e9385acd4b9759.html" :sig "-" :expr false}
"Clojure@range@core@8" {:lang "Clojure" :algo "range" :lib "core" :id 8 :doc "https://clojuredocs.org/clojure.core/range" :sig "-" :expr false}
"D@iota@range@8" {:lang "D" :algo "iota" :lib "range" :id 8 :doc "https://dlang.org/library/std/range/iota.html" :sig "-" :expr false}
"Excel@sequence@-@8" {:lang "Excel" :algo "sequence" :lib "-" :id 8 :doc "https://exceljet.net/functions/sequence-function" :sig "-" :expr false}
"Gleam@range@list@8" {:lang "Gleam" :algo "range" :lib "list" :id 8 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#range" :sig "-" :expr false}
"Go@iota@-@8" {:lang "Go" :algo "iota" :lib "-" :id 8 :doc "https://go.dev/ref/spec#Iota" :sig "-" :expr false}
"Haskell@range@Data.List.HT@8" {:lang "Haskell" :algo "range" :lib "Data.List.HT" :id 8 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:range" :sig "-" :expr false}
"J@i. (integers)@-@8" {:lang "J" :algo "i. (integers)" :lib "-" :id 8 :doc "https://code.jsoftware.com/wiki/Vocabulary/idot" :sig "-" :expr false}
"JavaScript@range@ramda@8" {:lang "JavaScript" :algo "range" :lib "ramda" :id 8 :doc "https://ramdajs.com/docs/#range" :sig "-" :expr false}
"Julia@range@Base@8" {:lang "Julia" :algo "range" :lib "Base" :id 8 :doc "https://docs.julialang.org/en/v1/base/math/#Base.range" :sig "-" :expr false}
"Mathematica@Range@-@8" {:lang "Mathematica" :algo "Range" :lib "-" :id 8 :doc "http://reference.wolfram.com/language/ref/Range.html" :sig "-" :expr false}
"Python@range@-@8" {:lang "Python" :algo "range" :lib "-" :id 8 :doc "https://docs.python.org/3/library/functions.html#func-range" :sig "-" :expr false}
"R@seq@-@8" {:lang "R" :algo "seq" :lib "-" :id 8 :doc "https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/seq" :sig "-" :expr false}
"Racket@range@base@8" {:lang "Racket" :algo "range" :lib "base" :id 8 :doc "https://docs.racket-lang.org/reference/pairs.html?q=enumerate#%28def._%28%28lib._racket%2Flist..rkt%29._range%29%29" :sig "-" :expr false}
"Rust@.. (range)@ops@8" {:lang "Rust" :algo ".. (range)" :lib "ops" :id 8 :doc "https://doc.rust-lang.org/std/ops/struct.Range.html" :sig "-" :expr false}
"Scala@until@Range@8" {:lang "Scala" :algo "until" :lib "Range" :id 8 :doc "https://scala-lang.org/api/current/scala/collection/immutable/Range.html" :sig "-" :expr false}
"Scheme@iota@list@8" {:lang "Scheme" :algo "iota" :lib "list" :id 8 :doc "https://index.scheme.org/filterset/r7rs_all/%28scheme%2520list%29/iota" :sig "-" :expr false}
"TinyAPL@⍳ (index generator)@-@8" {:lang "TinyAPL" :algo "⍳ (index generator)" :lib "-" :id 8 :doc "https://tinyapl.rubenverg.com/docs/primitive/index_generator" :sig "-" :expr false}
"Uiua@⇡ (range)@-@8" {:lang "Uiua" :algo "⇡ (range)" :lib "-" :id 8 :doc "https://www.uiua.org/docs/range" :sig "-" :expr false}
"k@! (enum)@-@8" {:lang "k" :algo "! (enum)" :lib "-" :id 8 :doc "https://ngn.codeberg.page/k/#" :sig "-" :expr false}
"q@til@-@8" {:lang "q" :algo "til" :lib "-" :id 8 :doc "https://code.kx.com/q/ref/til/" :sig "-" :expr false}
"☾@⭥  (indicies)@-@8" {:lang "☾" :algo "⭥  (indicies)" :lib "-" :id 8 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"☾@󷸹  (range not right)@-@8" {:lang "☾" :algo "󷸹  (range not right)" :lib "-" :id 8 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⍳∘≢,¨⊢@-@9" {:lang "APL" :algo "⍳∘≢,¨⊢" :lib "-" :id 9 :doc "" :sig "-" :expr true}
"BQN@↕∘≠⋈¨⊢@-@9" {:lang "BQN" :algo "↕∘≠⋈¨⊢" :lib "-" :id 9 :doc "" :sig "-" :expr true}
"C#@Select@Enumerable@9" {:lang "C#" :algo "Select" :lib "Enumerable" :id 9 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.select" :sig "-" :expr false}
"C++@enumerate@<ranges>@9" {:lang "C++" :algo "enumerate" :lib "<ranges>" :id 9 :doc "https://en.cppreference.com/w/cpp/ranges/enumerate_view" :sig "-" :expr false}
"Clojure@map-indexed*@core@9" {:lang "Clojure" :algo "map-indexed*" :lib "core" :id 9 :doc "https://clojuredocs.org/clojure.core/map-indexed" :sig "-" :expr false}
"D@enumerate@range@9" {:lang "D" :algo "enumerate" :lib "range" :id 9 :doc "https://dlang.org/library/std/range/enumerate.html" :sig "-" :expr false}
"Elixir@with_index@Enum@9" {:lang "Elixir" :algo "with_index" :lib "Enum" :id 9 :doc "https://hexdocs.pm/elixir/Enum.html#with_index/2" :sig "-" :expr false}
"F#@indexed@various@9" {:lang "F#" :algo "indexed" :lib "various" :id 9 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-arraymodule.html#indexed" :sig "-" :expr false}
"Haskell@indexed@Data.List.Index@9" {:lang "Haskell" :algo "indexed" :lib "Data.List.Index" :id 9 :doc "https://hackage.haskell.org/package/ilist-0.4.0.0/docs/Data-List-Index.html#v:indexed" :sig "-" :expr false}
"JavaScript@addIndex@ramda@9" {:lang "JavaScript" :algo "addIndex" :lib "ramda" :id 9 :doc "https://ramdajs.com/docs/#addIndex" :sig "-" :expr false}
"Julia@enumerate@Base@9" {:lang "Julia" :algo "enumerate" :lib "Base" :id 9 :doc "https://docs.julialang.org/en/v1/base/iterators/#Base.Iterators.enumerate" :sig "-" :expr false}
"Kotlin@withIndex@collections@9" {:lang "Kotlin" :algo "withIndex" :lib "collections" :id 9 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/with-index.html" :sig "-" :expr false}
"Python@enumerate@-@9" {:lang "Python" :algo "enumerate" :lib "-" :id 9 :doc "https://docs.python.org/3/library/functions.html#enumerate" :sig "-" :expr false}
"Racket@enumerate@list-utils@9" {:lang "Racket" :algo "enumerate" :lib "list-utils" :id 9 :doc "https://docs.racket-lang.org/list-utils/index.html?q=freq#%28def._%28%28lib._list-utils%2Fmain..rkt%29._enumerate%29%29" :sig "-" :expr false}
"Racket@indexed@data-collections@9" {:lang "Racket" :algo "indexed" :lib "data-collections" :id 9 :doc "https://docs.racket-lang.org/collections/collections-api.html?q=chunk-while#%28def._%28%28lib._data%2Fcollection..rkt%29._indexed%29%29" :sig "-" :expr false}
"Ruby@with_index@Enumerable@9" {:lang "Ruby" :algo "with_index" :lib "Enumerable" :id 9 :doc "https://apidock.com/ruby/Enumerator/with_index" :sig "-" :expr false}
"Rust@enumerate@iter@9" {:lang "Rust" :algo "enumerate" :lib "iter" :id 9 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.enumerate" :sig "-" :expr false}
"Scala@zipWithIndex@various@9" {:lang "Scala" :algo "zipWithIndex" :lib "various" :id 9 :doc "https://scala-lang.org/api/current/scala/collection/IterableOps.html#zipWithIndex-0" :sig "-" :expr false}
"Uiua@°⊏ (un select)@-@9" {:lang "Uiua" :algo "°⊏ (un select)" :lib "-" :id 9 :doc "https://www.uiua.org/docs/select" :sig "-" :expr true}
"☾@↨  (enumerate)@-@9" {:lang "☾" :algo "↨  (enumerate)" :lib "-" :id 9 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@partitionᵇ@Data.List@10" {:lang "Agda" :algo "partitionᵇ" :lib "Data.List" :id 10 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#11170" :sig "-" :expr false}
"C++@partition_copy@<algorithm>@10" {:lang "C++" :algo "partition_copy" :lib "<algorithm>" :id 10 :doc "https://en.cppreference.com/w/cpp/algorithm/partition_copy" :sig "-" :expr false}
"Elixir@split_with@Enum@10" {:lang "Elixir" :algo "split_with" :lib "Enum" :id 10 :doc "https://hexdocs.pm/elixir/Enum.html#split_with/2" :sig "-" :expr false}
"Haskell@partition@Data.List@10" {:lang "Haskell" :algo "partition" :lib "Data.List" :id 10 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Data-List.html#v:partition" :sig "-" :expr false}
"Julia@partition@Base@10" {:lang "Julia" :algo "partition" :lib "Base" :id 10 :doc "https://docs.julialang.org/en/v1/base/iterators/#Base.Iterators.partition" :sig "-" :expr false}
"Kotlin@partition@collections@10" {:lang "Kotlin" :algo "partition" :lib "collections" :id 10 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/partition.html" :sig "-" :expr false}
"Python@partition@more-itertools@10" {:lang "Python" :algo "partition" :lib "more-itertools" :id 10 :doc "https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partition" :sig "-" :expr false}
"Racket@partition@base@10" {:lang "Racket" :algo "partition" :lib "base" :id 10 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._partition%29%29" :sig "-" :expr false}
"Ruby@partition@Enumerable@10" {:lang "Ruby" :algo "partition" :lib "Enumerable" :id 10 :doc "https://apidock.com/ruby/Enumerable/partition" :sig "-" :expr false}
"Rust@partition@iter@10" {:lang "Rust" :algo "partition" :lib "iter" :id 10 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.partition" :sig "-" :expr false}
"Scala@partition@various@10" {:lang "Scala" :algo "partition" :lib "various" :id 10 :doc "https://scala-lang.org/api/current/scala/collection/Iterator.html#partition-243" :sig "-" :expr false}
"Uiua@⊕□ (group box)@-@10" {:lang "Uiua" :algo "⊕□ (group box)" :lib "-" :id 10 :doc "https://www.uiua.org/docs/group" :sig "-" :expr true}
"☾@􋇐 (bool groupby)@-@10" {:lang "☾" :algo "􋇐 (bool groupby)" :lib "-" :id 10 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"C++@chunk@<ranges>@11" {:lang "C++" :algo "chunk" :lib "<ranges>" :id 11 :doc "https://en.cppreference.com/w/cpp/ranges/chunk_view" :sig "-" :expr false}
"Clojure@partition@core@11" {:lang "Clojure" :algo "partition" :lib "core" :id 11 :doc "https://clojuredocs.org/clojure.core/partition" :sig "-" :expr false}
"D@chunks@std.range@11" {:lang "D" :algo "chunks" :lib "std.range" :id 11 :doc "https://dlang.org/library/std/range/chunks.html#0" :sig "-" :expr false}
"Go@Chunk@slices@11" {:lang "Go" :algo "Chunk" :lib "slices" :id 11 :doc "https://pkg.go.dev/slices#Chunk" :sig "-" :expr false}
"Haskell@chunksOf@Data.List.Split@11" {:lang "Haskell" :algo "chunksOf" :lib "Data.List.Split" :id 11 :doc "https://hackage.haskell.org/package/split-0.2.5/docs/Data-List-Split.html#v:chunksOf" :sig "-" :expr false}
"JavaScript@splitEvery@ramda@11" {:lang "JavaScript" :algo "splitEvery" :lib "ramda" :id 11 :doc "https://ramdajs.com/docs/#splitEvery" :sig "-" :expr false}
"Kotlin@chunked@collections@11" {:lang "Kotlin" :algo "chunked" :lib "collections" :id 11 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/chunked.html" :sig "-" :expr false}
"Mathematica@Partition@-@11" {:lang "Mathematica" :algo "Partition" :lib "-" :id 11 :doc "http://reference.wolfram.com/language/ref/Partition.html" :sig "-" :expr false}
"Rust@chunks@itertools@11" {:lang "Rust" :algo "chunks" :lib "itertools" :id 11 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.chunks" :sig "-" :expr false}
"Rust@chunks@-@11" {:lang "Rust" :algo "chunks" :lib "-" :id 11 :doc "https://doc.rust-lang.org/std/primitive.slice.html#method.chunks" :sig "-" :expr false}
"Scala@grouped@various@11" {:lang "Scala" :algo "grouped" :lib "various" :id 11 :doc "https://scala-lang.org/api/current/scala/collection/ArrayOps.html#grouped-fffffbef" :sig "-" :expr false}
"Uiua@⧈⌞∘ (stencil__< id)@-@11" {:lang "Uiua" :algo "⧈⌞∘ (stencil__< id)" :lib "-" :id 11 :doc "https://www.uiua.org/docs/stencil" :sig "-" :expr true}
"☾@ᙧ  (window)@-@11" {:lang "☾" :algo "ᙧ  (window)" :lib "-" :id 11 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@intersperse@Data.List@12" {:lang "Agda" :algo "intersperse" :lib "Data.List" :id 12 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#1819" :sig "-" :expr false}
"Elixir@intersperse@Enum@12" {:lang "Elixir" :algo "intersperse" :lib "Enum" :id 12 :doc "https://hexdocs.pm/elixir/1.12/Enum.html#intersperse/2" :sig "-" :expr false}
"Elm@intersperse@core@12" {:lang "Elm" :algo "intersperse" :lib "core" :id 12 :doc "https://package.elm-lang.org/packages/elm/core/latest/List#intersperse" :sig "-" :expr false}
"Gleam@intersperse@list@12" {:lang "Gleam" :algo "intersperse" :lib "list" :id 12 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#intersperse" :sig "-" :expr false}
"Haskell@intersperse@Data.List@12" {:lang "Haskell" :algo "intersperse" :lib "Data.List" :id 12 :doc "https://hackage.haskell.org/package/base/docs/Data-List.html#v:intersperse" :sig "-" :expr false}
"JavaScript@intersperse@ramda@12" {:lang "JavaScript" :algo "intersperse" :lib "ramda" :id 12 :doc "https://ramdajs.com/docs/#intersperse" :sig "-" :expr false}
"Racket@intersperse@seq@12" {:lang "Racket" :algo "intersperse" :lib "seq" :id 12 :doc "https://docs.racket-lang.org/seq/index.html#%28def._%28%28lib._seq%2Fmain..rkt%29._intersperse%29%29" :sig "-" :expr false}
"Racket@add-between@-@12" {:lang "Racket" :algo "add-between" :lib "-" :id 12 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._add-between%29%29" :sig "-" :expr false}
"Rust@intersperse@itertools@12" {:lang "Rust" :algo "intersperse" :lib "itertools" :id 12 :doc "https://docs.rs/itertools/latest/itertools/fn.intersperse.html" :sig "-" :expr false}
"☾@⨝ (join)@-@12" {:lang "☾" :algo "⨝ (join)" :lib "-" :id 12 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@group@Data.Vec@13" {:lang "Agda" :algo "group" :lib "Data.Vec" :id 13 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Vec.Base.html#7275" :sig "-" :expr false}
"C++@partition@<algorithm>@13" {:lang "C++" :algo "partition" :lib "<algorithm>" :id 13 :doc "https://en.cppreference.com/w/cpp/algorithm/partition" :sig "-" :expr false}
"CUDA@partition@Thrust@13" {:lang "CUDA" :algo "partition" :lib "Thrust" :id 13 :doc "https://thrust.github.io/doc/group__partitioning_gac5cdbb402c5473ca92e95bc73ecaf13c.html" :sig "-" :expr false}
"D@partition@algorithm.sorting@13" {:lang "D" :algo "partition" :lib "algorithm.sorting" :id 13 :doc "https://dlang.org/library/std/algorithm/sorting/partition.html" :sig "-" :expr false}
"JavaScript@partition@ramda@13" {:lang "JavaScript" :algo "partition" :lib "ramda" :id 13 :doc "https://ramdajs.com/docs/#partition" :sig "-" :expr false}
"Uiua@⊏⍖ (select fall)@-@13" {:lang "Uiua" :algo "⊏⍖ (select fall)" :lib "-" :id 13 :doc "https://www.uiua.org/docs/fall" :sig "-" :expr true}
"☾@󰒽  (sort decreasing)@-@13" {:lang "☾" :algo "󰒽  (sort decreasing)" :lib "-" :id 13 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"C++@unique@<algorithm>@14" {:lang "C++" :algo "unique" :lib "<algorithm>" :id 14 :doc "https://en.cppreference.com/w/cpp/algorithm/unique" :sig "-" :expr false}
"D@uniq@algorithm.iteration@14" {:lang "D" :algo "uniq" :lib "algorithm.iteration" :id 14 :doc "https://dlang.org/phobos/std_algorithm_iteration.html#uniq" :sig "-" :expr false}
"Elm@dedup@List@14" {:lang "Elm" :algo "dedup" :lib "List" :id 14 :doc "https://package.elm-lang.org/packages/mgold/elm-nonempty-list/latest/List-Nonempty#dedup" :sig "-" :expr false}
"Go@Compact@slices@14" {:lang "Go" :algo "Compact" :lib "slices" :id 14 :doc "https://pkg.go.dev/slices#Compact" :sig "-" :expr false}
"JavaScript@uniq@ramda@14" {:lang "JavaScript" :algo "uniq" :lib "ramda" :id 14 :doc "https://ramdajs.com/docs/#uniq" :sig "-" :expr false}
"Mathematica@DeleteAdjacentDuplicates@-@14" {:lang "Mathematica" :algo "DeleteAdjacentDuplicates" :lib "-" :id 14 :doc "http://reference.wolfram.com/language/ref/DeleteAdjacentDuplicates.html" :sig "-" :expr false}
"Rust@dedup@itertools@14" {:lang "Rust" :algo "dedup" :lib "itertools" :id 14 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.dedup" :sig "-" :expr false}
"Shell@uniq@-@14" {:lang "Shell" :algo "uniq" :lib "-" :id 14 :doc "https://ss64.com/bash/uniq.html" :sig "-" :expr false}
"Uiua@◌°▽ (pop un keep)@-@14" {:lang "Uiua" :algo "◌°▽ (pop un keep)" :lib "-" :id 14 :doc "https://www.uiua.org/docs/keep" :sig "-" :expr true}
"APL@∪ (unique)@-@15" {:lang "APL" :algo "∪ (unique)" :lib "-" :id 15 :doc "https://aplwiki.com/wiki/Unique" :sig "-" :expr false}
"Agda@deduplicate@Data.List@15" {:lang "Agda" :algo "deduplicate" :lib "Data.List" :id 15 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#13320" :sig "-" :expr false}
"BQN@⍷ (deduplicate)@-@15" {:lang "BQN" :algo "⍷ (deduplicate)" :lib "-" :id 15 :doc "https://mlochbaum.github.io/BQN/doc/selfcmp.html#deduplicate" :sig "-" :expr false}
"C#@Distinct@Enumerable@15" {:lang "C#" :algo "Distinct" :lib "Enumerable" :id 15 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.distinct" :sig "-" :expr false}
"Clojure@distinct@core@15" {:lang "Clojure" :algo "distinct" :lib "core" :id 15 :doc "https://clojuredocs.org/clojure.core/distinct" :sig "-" :expr false}
"Common Lisp@remove-duplicates@-@15" {:lang "Common Lisp" :algo "remove-duplicates" :lib "-" :id 15 :doc "https://novaspec.org/cl/f_remove-duplicates" :sig "-" :expr false}
"Elixir@uniq@Enum@15" {:lang "Elixir" :algo "uniq" :lib "Enum" :id 15 :doc "https://hexdocs.pm/elixir/Enum.html#uniq/1" :sig "-" :expr false}
"Elm@uniq@List@15" {:lang "Elm" :algo "uniq" :lib "List" :id 15 :doc "https://package.elm-lang.org/packages/mgold/elm-nonempty-list/latest/List-Nonempty#uniq" :sig "-" :expr false}
"Excel@unique@-@15" {:lang "Excel" :algo "unique" :lib "-" :id 15 :doc "https://support.microsoft.com/en-us/office/unique-function-c5ab87fd-30a3-4ce9-9d1a-40204fb85e1e" :sig "-" :expr false}
"F#@distinct@Seq@15" {:lang "F#" :algo "distinct" :lib "Seq" :id 15 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-seqmodule.html#distinct" :sig "-" :expr false}
"Haskell@nub@Data.List@15" {:lang "Haskell" :algo "nub" :lib "Data.List" :id 15 :doc "https://hackage.haskell.org/package/base-4.18.1.0/docs/Data-List.html#v:nub" :sig "-" :expr false}
"Haskell@sortUniq@Data.List.Unique@15" {:lang "Haskell" :algo "sortUniq" :lib "Data.List.Unique" :id 15 :doc "https://hackage.haskell.org/package/Unique-0.4.7.7/docs/Data-List-Unique.html#v:sortUniq" :sig "-" :expr false}
"J@~. (nub)@-@15" {:lang "J" :algo "~. (nub)" :lib "-" :id 15 :doc "https://code.jsoftware.com/wiki/Vocabulary/tildedot" :sig "-" :expr false}
"JavaScript@uniq@ramda@15" {:lang "JavaScript" :algo "uniq" :lib "ramda" :id 15 :doc "https://ramdajs.com/docs/#uniq" :sig "-" :expr false}
"Julia@unique@Base@15" {:lang "Julia" :algo "unique" :lib "Base" :id 15 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.unique" :sig "-" :expr false}
"Kap@∪ (unique)@-@15" {:lang "Kap" :algo "∪ (unique)" :lib "-" :id 15 :doc "" :sig "-" :expr false}
"Kotlin@distinct@collections@15" {:lang "Kotlin" :algo "distinct" :lib "collections" :id 15 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/distinct.html" :sig "-" :expr false}
"Mathematica@DeleteDuplicates@-@15" {:lang "Mathematica" :algo "DeleteDuplicates" :lib "-" :id 15 :doc "http://reference.wolfram.com/language/ref/DeleteDuplicates.html" :sig "-" :expr false}
"OCaml@sort_uniq@List@15" {:lang "OCaml" :algo "sort_uniq" :lib "List" :id 15 :doc "https://ocaml.org/api/List.html#VALsort_uniq" :sig "-" :expr false}
"Pharo@removeDuplicates@OrderedCollection@15" {:lang "Pharo" :algo "removeDuplicates" :lib "OrderedCollection" :id 15 :doc "" :sig "-" :expr false}
"Racket@remove-duplicates@-@15" {:lang "Racket" :algo "remove-duplicates" :lib "-" :id 15 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket/list..rkt%29._remove-duplicates%29%29" :sig "-" :expr false}
"Ruby@uniq@Array@15" {:lang "Ruby" :algo "uniq" :lib "Array" :id 15 :doc "https://apidock.com/ruby/Array/uniq" :sig "-" :expr false}
"Rust@unique@itertools@15" {:lang "Rust" :algo "unique" :lib "itertools" :id 15 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.unique" :sig "-" :expr false}
"Scala@distinct@various@15" {:lang "Scala" :algo "distinct" :lib "various" :id 15 :doc "https://scala-lang.org/api/current/scala/collection/SeqOps.html#distinct-0" :sig "-" :expr false}
"TinyAPL@∪ (unique)@-@15" {:lang "TinyAPL" :algo "∪ (unique)" :lib "-" :id 15 :doc "https://tinyapl.rubenverg.com/docs/primitive/unique" :sig "-" :expr false}
"Uiua@◴ (deduplicate)@-@15" {:lang "Uiua" :algo "◴ (deduplicate)" :lib "-" :id 15 :doc "https://www.uiua.org/docs/deduplicate" :sig "-" :expr false}
"☾@ᚤ  (unique)@-@15" {:lang "☾" :algo "ᚤ  (unique)" :lib "-" :id 15 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⍕ (format)@-@16" {:lang "APL" :algo "⍕ (format)" :lib "-" :id 16 :doc "http://microapl.com/apl_help/ch_020_020_680.htm" :sig "-" :expr false}
"Agda@show@various@16" {:lang "Agda" :algo "show" :lib "various" :id 16 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Bool.Show.html#412" :sig "-" :expr false}
"C++@to_string@<string>@16" {:lang "C++" :algo "to_string" :lib "<string>" :id 16 :doc "https://en.cppreference.com/w/cpp/string/basic_string/to_string" :sig "-" :expr false}
"Clojure@str@core@16" {:lang "Clojure" :algo "str" :lib "core" :id 16 :doc "https://clojuredocs.org/clojure.core/str" :sig "-" :expr false}
"Elixir@to_string@various@16" {:lang "Elixir" :algo "to_string" :lib "various" :id 16 :doc "https://hexdocs.pm/elixir/Integer.html#to_string/1" :sig "-" :expr false}
"Excel@arraytotext@-@16" {:lang "Excel" :algo "arraytotext" :lib "-" :id 16 :doc "https://support.microsoft.com/en-us/office/arraytotext-function-9cdcad46-2fa5-4c6b-ac92-14e7bc862b8b" :sig "-" :expr false}
"Excel@textjoin@-@16" {:lang "Excel" :algo "textjoin" :lib "-" :id 16 :doc "https://support.microsoft.com/en-us/office/textjoin-function-357b449a-ec91-49d0-80c3-0e8fc845691c" :sig "-" :expr false}
"Haskell@show@Prelude@16" {:lang "Haskell" :algo "show" :lib "Prelude" :id 16 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:show" :sig "-" :expr false}
"J@\": (default format)@-@16" {:lang "J" :algo "\": (default format)" :lib "-" :id 16 :doc "https://code.jsoftware.com/wiki/Vocabulary/quoteco" :sig "-" :expr false}
"JavaScript@toString@-@16" {:lang "JavaScript" :algo "toString" :lib "-" :id 16 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/toString" :sig "-" :expr false}
"JavaScript@toString@ramda@16" {:lang "JavaScript" :algo "toString" :lib "ramda" :id 16 :doc "https://ramdajs.com/docs/#toString" :sig "-" :expr false}
"Julia@string@Base@16" {:lang "Julia" :algo "string" :lib "Base" :id 16 :doc "https://docs.julialang.org/en/v1/base/strings/#Base.string" :sig "-" :expr false}
"Mathematica@ToString@-@16" {:lang "Mathematica" :algo "ToString" :lib "-" :id 16 :doc "http://reference.wolfram.com/language/ref/ToString.html" :sig "-" :expr false}
"Python@str@-@16" {:lang "Python" :algo "str" :lib "-" :id 16 :doc "https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str" :sig "-" :expr false}
"Ruby@to_s@-@16" {:lang "Ruby" :algo "to_s" :lib "-" :id 16 :doc "https://apidock.com/ruby/Object/to_s" :sig "-" :expr false}
"Rust@to_string@string@16" {:lang "Rust" :algo "to_string" :lib "string" :id 16 :doc "https://doc.rust-lang.org/std/string/trait.ToString.html#tymethod.to_string" :sig "-" :expr false}
"Scala@toString@various@16" {:lang "Scala" :algo "toString" :lib "various" :id 16 :doc "https://scala-lang.org/api/current/scala/reflect/AnyValManifest.html#toString-0" :sig "-" :expr false}
"TinyAPL@⍕ (format)@-@16" {:lang "TinyAPL" :algo "⍕ (format)" :lib "-" :id 16 :doc "https://tinyapl.rubenverg.com/docs/primitive/format" :sig "-" :expr false}
"q@string@-@16" {:lang "q" :algo "string" :lib "-" :id 16 :doc "https://code.kx.com/v2/ref/string/" :sig "-" :expr false}
"☾@ᔐ  (str)@-@16" {:lang "☾" :algo "ᔐ  (str)" :lib "-" :id 16 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@_∈_@Data.List@17" {:lang "Agda" :algo "_∈_" :lib "Data.List" :id 17 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Membership.Setoid.html#925" :sig "-" :expr false}
"C++@contains@algorithm@17" {:lang "C++" :algo "contains" :lib "algorithm" :id 17 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/contains" :sig "-" :expr false}
"Clojure@contains?@core@17" {:lang "Clojure" :algo "contains?" :lib "core" :id 17 :doc "https://clojuredocs.org/clojure.core/contains_q" :sig "-" :expr false}
"D@canFind@algorithm.searching@17" {:lang "D" :algo "canFind" :lib "algorithm.searching" :id 17 :doc "https://dlang.org/library/std/algorithm/searching/can_find.can_find.html" :sig "-" :expr false}
"Elixir@member?@Enum@17" {:lang "Elixir" :algo "member?" :lib "Enum" :id 17 :doc "https://hexdocs.pm/elixir/Enum.html#member?/2" :sig "-" :expr false}
"Excel@= (equal)@-@17" {:lang "Excel" :algo "= (equal)" :lib "-" :id 17 :doc "" :sig "-" :expr false}
"Go@Contains@slices@17" {:lang "Go" :algo "Contains" :lib "slices" :id 17 :doc "https://pkg.go.dev/slices#Contains" :sig "-" :expr false}
"Haskell@elem@Prelude@17" {:lang "Haskell" :algo "elem" :lib "Prelude" :id 17 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:elem" :sig "-" :expr false}
"JavaScript@includes@-@17" {:lang "JavaScript" :algo "includes" :lib "-" :id 17 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/includes" :sig "-" :expr false}
"JavaScript@includes@ramda@17" {:lang "JavaScript" :algo "includes" :lib "ramda" :id 17 :doc "https://ramdajs.com/docs/#includes" :sig "-" :expr false}
"Julia@in@Base@17" {:lang "Julia" :algo "in" :lib "Base" :id 17 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.in" :sig "-" :expr false}
"Julia@contains@Base@17" {:lang "Julia" :algo "contains" :lib "Base" :id 17 :doc "https://docs.julialang.org/en/v1/base/strings/#Base.contains" :sig "-" :expr false}
"Kotlin@contains@collections@17" {:lang "Kotlin" :algo "contains" :lib "collections" :id 17 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/contains.html" :sig "-" :expr false}
"Mathematica@MemberQ@-@17" {:lang "Mathematica" :algo "MemberQ" :lib "-" :id 17 :doc "http://reference.wolfram.com/language/ref/MemberQ.html" :sig "-" :expr false}
"Python@in@-@17" {:lang "Python" :algo "in" :lib "-" :id 17 :doc "https://www.programiz.com/python-programming/keyword-list#in" :sig "-" :expr false}
"Racket@member*@base@17" {:lang "Racket" :algo "member*" :lib "base" :id 17 :doc "https://docs.racket-lang.org/reference/pairs.html?q=map#%28def._%28%28lib._racket%2Fprivate%2Fbase..rkt%29._member%29%29" :sig "-" :expr false}
"Ruby@include?@various@17" {:lang "Ruby" :algo "include?" :lib "various" :id 17 :doc "https://apidock.com/ruby/Array/include%3F" :sig "-" :expr false}
"Scala@contains@various@17" {:lang "Scala" :algo "contains" :lib "various" :id 17 :doc "https://scala-lang.org/api/current/scala/collection/Iterator.html#contains-4ad" :sig "-" :expr false}
"TinyAPL@∊ (element of)@-@17" {:lang "TinyAPL" :algo "∊ (element of)" :lib "-" :id 17 :doc "https://tinyapl.rubenverg.com/docs/primitive/element_of" :sig "-" :expr false}
"Uiua@∊ (memberof)@-@17" {:lang "Uiua" :algo "∊ (memberof)" :lib "-" :id 17 :doc "http://uiua.org/docs/memberof" :sig "-" :expr false}
"☾@∈  (left in right)@-@17" {:lang "☾" :algo "∈  (left in right)" :lib "-" :id 17 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"☾@∋  (right in left)@-@17" {:lang "☾" :algo "∋  (right in left)" :lib "-" :id 17 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⍸ (where)@-@18" {:lang "APL" :algo "⍸ (where)" :lib "-" :id 18 :doc "https://help.dyalog.com/17.1/#Language/Primitive%20Functions/Where.htm#kanchor3608" :sig "-" :expr false}
"BQN@/ (indices)@-@18" {:lang "BQN" :algo "/ (indices)" :lib "-" :id 18 :doc "https://mlochbaum.github.io/BQN/doc/replicate.html#indices" :sig "-" :expr false}
"J@I. (indices)@-@18" {:lang "J" :algo "I. (indices)" :lib "-" :id 18 :doc "https://code.jsoftware.com/wiki/Vocabulary/icapdot" :sig "-" :expr false}
"JavaScript@where@ramda@18" {:lang "JavaScript" :algo "where" :lib "ramda" :id 18 :doc "https://ramdajs.com/docs/#where" :sig "-" :expr false}
"Julia@findall@Base@18" {:lang "Julia" :algo "findall" :lib "Base" :id 18 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.findall-Tuple{Function,%20Any}" :sig "-" :expr false}
"Kap@⍸ (where)@-@18" {:lang "Kap" :algo "⍸ (where)" :lib "-" :id 18 :doc "" :sig "-" :expr false}
"Python@where@NumPy@18" {:lang "Python" :algo "where" :lib "NumPy" :id 18 :doc "https://numpy.org/doc/stable/reference/generated/numpy.where.html" :sig "-" :expr false}
"TinyAPL@⍸ (where)@-@18" {:lang "TinyAPL" :algo "⍸ (where)" :lib "-" :id 18 :doc "https://tinyapl.rubenverg.com/docs/primitive/where" :sig "-" :expr false}
"Uiua@⊚ (where)@-@18" {:lang "Uiua" :algo "⊚ (where)" :lib "-" :id 18 :doc "https://www.uiua.org/docs/where" :sig "-" :expr false}
"k@& (where)@-@18" {:lang "k" :algo "& (where)" :lib "-" :id 18 :doc "https://ngn.codeberg.page/k/#" :sig "-" :expr false}
"q@where@-@18" {:lang "q" :algo "where" :lib "-" :id 18 :doc "https://code.kx.com/q4m3/A_Built-in_Functions/#a107-where" :sig "-" :expr false}
"☾@󰈲ᔨ (filter correspondence)@-@18" {:lang "☾" :algo "󰈲ᔨ (filter correspondence)" :lib "-" :id 18 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Clojure@frequencies@core@19" {:lang "Clojure" :algo "frequencies" :lib "core" :id 19 :doc "https://clojuredocs.org/clojure.core/frequencies" :sig "-" :expr false}
"Haskell@count@Data.List.Unique@19" {:lang "Haskell" :algo "count" :lib "Data.List.Unique" :id 19 :doc "https://hackage.haskell.org/package/Unique-0.4.7.7/docs/Data-List-Unique.html#v:count" :sig "-" :expr false}
"JavaScript@count@ramda@19" {:lang "JavaScript" :algo "count" :lib "ramda" :id 19 :doc "https://ramdajs.com/docs/#count" :sig "-" :expr false}
"Mathematica@Counts@-@19" {:lang "Mathematica" :algo "Counts" :lib "-" :id 19 :doc "http://reference.wolfram.com/language/ref/Counts.html" :sig "-" :expr false}
"Python@Counter*@collections@19" {:lang "Python" :algo "Counter*" :lib "collections" :id 19 :doc "https://docs.python.org/3/library/collections.html#collections.Counter" :sig "-" :expr false}
"Python@value_counts@RAPIDS cuDF@19" {:lang "Python" :algo "value_counts" :lib "RAPIDS cuDF" :id 19 :doc "https://docs.rapids.ai/api/cudf/stable/api.html#cudf.core.series.Series.value_counts" :sig "-" :expr false}
"Python@value_counts@pandas@19" {:lang "Python" :algo "value_counts" :lib "pandas" :id 19 :doc "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html" :sig "-" :expr false}
"Racket@frequencies@list-utils@19" {:lang "Racket" :algo "frequencies" :lib "list-utils" :id 19 :doc "https://docs.racket-lang.org/list-utils/index.html?q=freq#%28def._%28%28lib._list-utils%2Fmain..rkt%29._frequencies%29%29" :sig "-" :expr false}
"Rust@counts@itertools@19" {:lang "Rust" :algo "counts" :lib "itertools" :id 19 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.counts" :sig "-" :expr false}
"Uiua@°⊚⊛ (un where classify)@-@19" {:lang "Uiua" :algo "°⊚⊛ (un where classify)" :lib "-" :id 19 :doc "https://www.uiua.org/docs/classify" :sig "-" :expr true}
"Agda@any@various@20" {:lang "Agda" :algo "any" :lib "various" :id 20 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#4857" :sig "-" :expr false}
"C++@any_of@<algorithm>@20" {:lang "C++" :algo "any_of" :lib "<algorithm>" :id 20 :doc "https://en.cppreference.com/w/cpp/algorithm/all_any_none_of" :sig "-" :expr false}
"Clojure@some@core@20" {:lang "Clojure" :algo "some" :lib "core" :id 20 :doc "https://clojuredocs.org/clojure.core/some" :sig "-" :expr false}
"D@any@algorithm.searching@20" {:lang "D" :algo "any" :lib "algorithm.searching" :id 20 :doc "https://dlang.org/library/std/algorithm/searching/any.html" :sig "-" :expr false}
"Elixir@any?@Enum@20" {:lang "Elixir" :algo "any?" :lib "Enum" :id 20 :doc "https://hexdocs.pm/elixir/Enum.html#any?/2" :sig "-" :expr false}
"Erlang@any@lists@31" {:lang "Erlang" :algo "any" :lib "lists" :id 31 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#any/2" :sig "-" :expr false}
"F#@exists@List@20" {:lang "F#" :algo "exists" :lib "List" :id 20 :doc "https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/lists" :sig "-" :expr false}
"Haskell@any@Prelude@20" {:lang "Haskell" :algo "any" :lib "Prelude" :id 20 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:any" :sig "-" :expr false}
"JavaScript@some@-@20" {:lang "JavaScript" :algo "some" :lib "-" :id 20 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/some" :sig "-" :expr false}
"JavaScript@any@ramda@20" {:lang "JavaScript" :algo "any" :lib "ramda" :id 20 :doc "https://ramdajs.com/docs/#any" :sig "-" :expr false}
"Julia@any@Base@20" {:lang "Julia" :algo "any" :lib "Base" :id 20 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.any-Tuple%7BAny%7D" :sig "-" :expr false}
"Kotlin@any@collections@20" {:lang "Kotlin" :algo "any" :lib "collections" :id 20 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/any.html" :sig "-" :expr false}
"Mathematica@AnyTrue@-@20" {:lang "Mathematica" :algo "AnyTrue" :lib "-" :id 20 :doc "http://reference.wolfram.com/language/ref/AnyTrue.html" :sig "-" :expr false}
"Python@any@-@20" {:lang "Python" :algo "any" :lib "-" :id 20 :doc "https://docs.python.org/3/library/functions.html#any" :sig "-" :expr false}
"Racket@memf*@base@20" {:lang "Racket" :algo "memf*" :lib "base" :id 20 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Fprivate%2Flist..rkt%29._memf%29%29" :sig "-" :expr false}
"Ruby@any?@Enumerable@20" {:lang "Ruby" :algo "any?" :lib "Enumerable" :id 20 :doc "https://apidock.com/ruby/Enumerable/any%3F" :sig "-" :expr false}
"Rust@any@iter@20" {:lang "Rust" :algo "any" :lib "iter" :id 20 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.any" :sig "-" :expr false}
"Scala@exists@various@20" {:lang "Scala" :algo "exists" :lib "various" :id 20 :doc "https://scala-lang.org/api/current/scala/collection/IterableOnceOps.html#exists-fffff66f" :sig "-" :expr false}
"Uiua@/↥ (reduce maximum)@-@20" {:lang "Uiua" :algo "/↥ (reduce maximum)" :lib "-" :id 20 :doc "https://www.uiua.org/docs/reduce" :sig "-" :expr true}
"☾@⋁  (any)@-@20" {:lang "☾" :algo "⋁  (any)" :lib "-" :id 20 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@, (catenate)@-@21" {:lang "APL" :algo ", (catenate)" :lib "-" :id 21 :doc "http://microapl.com/apl_help/ch_020_020_490.htm" :sig "-" :expr false}
"Agda@concat@various@21" {:lang "Agda" :algo "concat" :lib "various" :id 21 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#4350" :sig "-" :expr false}
"C#@join@String@21" {:lang "C#" :algo "join" :lib "String" :id 21 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.string.join?view=netframework-4.8" :sig "-" :expr false}
"C++@join@<ranges>@21" {:lang "C++" :algo "join" :lib "<ranges>" :id 21 :doc "https://en.cppreference.com/w/cpp/ranges/join_view" :sig "-" :expr false}
"Clojure@join@string@21" {:lang "Clojure" :algo "join" :lib "string" :id 21 :doc "https://clojuredocs.org/clojure.string/join" :sig "-" :expr false}
"D@join@array@21" {:lang "D" :algo "join" :lib "array" :id 21 :doc "https://dlang.org/library/std/array/join.html" :sig "-" :expr false}
"Excel@textjoin@-@21" {:lang "Excel" :algo "textjoin" :lib "-" :id 21 :doc "https://support.microsoft.com/en-us/office/textjoin-function-357b449a-ec91-49d0-80c3-0e8fc845691c" :sig "-" :expr false}
"F#@join@String@21" {:lang "F#" :algo "join" :lib "String" :id 21 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.string.join?view=netframework-4.8" :sig "-" :expr false}
"Go@Join@strings@21" {:lang "Go" :algo "Join" :lib "strings" :id 21 :doc "https://golang.org/pkg/strings/#Join" :sig "-" :expr false}
"Go@Concat@slices@21" {:lang "Go" :algo "Concat" :lib "slices" :id 21 :doc "https://pkg.go.dev/slices#Concat" :sig "-" :expr false}
"Haskell@concat@Prelude@21" {:lang "Haskell" :algo "concat" :lib "Prelude" :id 21 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:concat" :sig "-" :expr false}
"JavaScript@join@-@21" {:lang "JavaScript" :algo "join" :lib "-" :id 21 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/join" :sig "-" :expr false}
"JavaScript@join@ramda@21" {:lang "JavaScript" :algo "join" :lib "ramda" :id 21 :doc "https://ramdajs.com/docs/#join" :sig "-" :expr false}
"Julia@join@Base@21" {:lang "Julia" :algo "join" :lib "Base" :id 21 :doc "https://docs.julialang.org/en/v1/base/strings/#Base.join" :sig "-" :expr false}
"Kotlin@joinTo@collections@21" {:lang "Kotlin" :algo "joinTo" :lib "collections" :id 21 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/join-to.html" :sig "-" :expr false}
"Mathematica@Join@-@21" {:lang "Mathematica" :algo "Join" :lib "-" :id 21 :doc "http://reference.wolfram.com/language/ref/Join.html" :sig "-" :expr false}
"Mathematica@StringJoin@-@21" {:lang "Mathematica" :algo "StringJoin" :lib "-" :id 21 :doc "http://reference.wolfram.com/language/ref/StringJoin.html" :sig "-" :expr false}
"Mathematica@Catenate@-@21" {:lang "Mathematica" :algo "Catenate" :lib "-" :id 21 :doc "http://reference.wolfram.com/language/ref/Catenate.html" :sig "-" :expr false}
"Python@join*@str@21" {:lang "Python" :algo "join*" :lib "str" :id 21 :doc "https://docs.python.org/3/library/stdtypes.html#str.join" :sig "-" :expr false}
"Racket@join@base@21" {:lang "Racket" :algo "join" :lib "base" :id 21 :doc "https://docs.racket-lang.org/algebraic/class_base.html?q=some#%28def._%28%28lib._algebraic%2Fcontrol%2Fmonad..rkt%29._join%29%29" :sig "-" :expr false}
"Ruby@join@various@21" {:lang "Ruby" :algo "join" :lib "various" :id 21 :doc "https://apidock.com/ruby/Array/join" :sig "-" :expr false}
"Rust@join@vec@21" {:lang "Rust" :algo "join" :lib "vec" :id 21 :doc "https://doc.rust-lang.org/std/vec/struct.Vec.html#method.join" :sig "-" :expr false}
"Rust@concat@vec@21" {:lang "Rust" :algo "concat" :lib "vec" :id 21 :doc "https://doc.rust-lang.org/std/vec/struct.Vec.html#method.concat" :sig "-" :expr false}
"Scala@flatten@various@21" {:lang "Scala" :algo "flatten" :lib "various" :id 21 :doc "https://scala-lang.org/api/current/scala/collection/IterableOnceOps.html#flatten-5d3" :sig "-" :expr false}
"Scala@mkString@various@21" {:lang "Scala" :algo "mkString" :lib "various" :id 21 :doc "https://scala-lang.org/api/current/scala/collection/IterableOnceExtensionMethods.html#mkString-0" :sig "-" :expr false}
"TinyAPL@⍪ (join)@-@21" {:lang "TinyAPL" :algo "⍪ (join)" :lib "-" :id 21 :doc "https://tinyapl.rubenverg.com/docs/primitive/join" :sig "-" :expr false}
"Uiua@♭ (deshape)@-@21" {:lang "Uiua" :algo "♭ (deshape)" :lib "-" :id 21 :doc "https://www.uiua.org/docs/deshape" :sig "-" :expr false}
"☾@⨝  (join)@-@21" {:lang "☾" :algo "⨝  (join)" :lib "-" :id 21 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⍉ (transpose)@-@22" {:lang "APL" :algo "⍉ (transpose)" :lib "-" :id 22 :doc "http://microapl.com/apl_help/ch_020_020_540.htm" :sig "-" :expr false}
"Agda@transpose@various@22" {:lang "Agda" :algo "transpose" :lib "various" :id 22 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Vec.Base.html#9519" :sig "-" :expr false}
"Clojure@transpose@core.matrix@22" {:lang "Clojure" :algo "transpose" :lib "core.matrix" :id 22 :doc "https://mikera.github.io/core.matrix/doc/clojure.core.matrix.html#var-transpose" :sig "-" :expr false}
"D@transposed@range@22" {:lang "D" :algo "transposed" :lib "range" :id 22 :doc "https://dlang.org/library/std/range/transposed.html" :sig "-" :expr false}
"Elixir@transpose@Matrix@22" {:lang "Elixir" :algo "transpose" :lib "Matrix" :id 22 :doc "https://hexdocs.pm/matrix/Matrix.html#transpose/1" :sig "-" :expr false}
"Excel@transpose@-@22" {:lang "Excel" :algo "transpose" :lib "-" :id 22 :doc "https://support.microsoft.com/en-us/office/transpose-function-ed039415-ed8a-4a81-93e9-4b6dfac76027" :sig "-" :expr false}
"F#@transpose@List@22" {:lang "F#" :algo "transpose" :lib "List" :id 22 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-listmodule.html#transpose" :sig "-" :expr false}
"Haskell@transpose@Data.List@22" {:lang "Haskell" :algo "transpose" :lib "Data.List" :id 22 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Data-List.html#v:transpose" :sig "-" :expr false}
"J@|: (transpose)@-@22" {:lang "J" :algo "|: (transpose)" :lib "-" :id 22 :doc "https://code.jsoftware.com/wiki/Vocabulary/barco" :sig "-" :expr false}
"Julia@transpose@LinearAlgebra@22" {:lang "Julia" :algo "transpose" :lib "LinearAlgebra" :id 22 :doc "https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.transpose" :sig "-" :expr false}
"Mathematica@Transpose@-@22" {:lang "Mathematica" :algo "Transpose" :lib "-" :id 22 :doc "http://reference.wolfram.com/language/ref/Transpose.html" :sig "-" :expr false}
"Python@zip(*)@-@22" {:lang "Python" :algo "zip(*)" :lib "-" :id 22 :doc "https://docs.python.org/3/library/functions.html#zip" :sig "-" :expr true}
"Python@transpose@pandas@22" {:lang "Python" :algo "transpose" :lib "pandas" :id 22 :doc "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html" :sig "-" :expr false}
"Python@transpose@RAPIDS cuDF@22" {:lang "Python" :algo "transpose" :lib "RAPIDS cuDF" :id 22 :doc "https://docs.rapids.ai/api/cudf/stable/api.html#cudf.core.dataframe.DataFrame.transpose" :sig "-" :expr false}
"Ruby@transpose@Array@22" {:lang "Ruby" :algo "transpose" :lib "Array" :id 22 :doc "https://apidock.com/ruby/Array/transpose" :sig "-" :expr false}
"Scala@transpose@various@22" {:lang "Scala" :algo "transpose" :lib "various" :id 22 :doc "https://scala-lang.org/api/current/scala/collection/ArrayOps.html#transpose-fffff875" :sig "-" :expr false}
"TinyAPL@⍉ (transpose)@-@22" {:lang "TinyAPL" :algo "⍉ (transpose)" :lib "-" :id 22 :doc "https://tinyapl.rubenverg.com/docs/primitive/transpose" :sig "-" :expr false}
"Uiua@⍉ (transpose)@-@22" {:lang "Uiua" :algo "⍉ (transpose)" :lib "-" :id 22 :doc "https://www.uiua.org/docs/transpose" :sig "-" :expr false}
"q@flip@-@22" {:lang "q" :algo "flip" :lib "-" :id 22 :doc "https://code.kx.com/v2/ref/flip/" :sig "-" :expr false}
"☾@⍉  (transpose)@-@22" {:lang "☾" :algo "⍉  (transpose)" :lib "-" :id 22 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"☾@󱑼  (zip)@-@22" {:lang "☾" :algo "󱑼  (zip)" :lib "-" :id 22 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⊆ (partition)@-@23" {:lang "APL" :algo "⊆ (partition)" :lib "-" :id 23 :doc "http://microapl.com/apl_help/ch_020_020_590.htm" :sig "-" :expr false}
"C++@chunk_by@range-v3@23" {:lang "C++" :algo "chunk_by" :lib "range-v3" :id 23 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-" :expr false}
"C++@chunk_by@<ranges>@23" {:lang "C++" :algo "chunk_by" :lib "<ranges>" :id 23 :doc "https://en.cppreference.com/w/cpp/ranges/chunk_by_view" :sig "-" :expr false}
"Clojure@partition-by@core@23" {:lang "Clojure" :algo "partition-by" :lib "core" :id 23 :doc "https://clojuredocs.org/clojure.core/partition-by" :sig "-" :expr false}
"D@chunkBy@algorith.iteration@23" {:lang "D" :algo "chunkBy" :lib "algorith.iteration" :id 23 :doc "https://dlang.org/library/std/algorithm/iteration/chunk_by.html" :sig "-" :expr false}
"Elixir@chunk_while@Enum@23" {:lang "Elixir" :algo "chunk_while" :lib "Enum" :id 23 :doc "https://hexdocs.pm/elixir/Enum.html#chunk_while/4" :sig "-" :expr false}
"Excel@groupby@-@23" {:lang "Excel" :algo "groupby" :lib "-" :id 23 :doc "https://support.microsoft.com/en-us/office/groupby-function-5e08ae8c-6800-4b72-b623-c41773611505" :sig "-" :expr false}
"Haskell@groupBy@Data.List.HT@23" {:lang "Haskell" :algo "groupBy" :lib "Data.List.HT" :id 23 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:groupBy" :sig "-" :expr false}
"JavaScript@splitWhen@ramda@23" {:lang "JavaScript" :algo "splitWhen" :lib "ramda" :id 23 :doc "https://ramdajs.com/docs/#splitWhen" :sig "-" :expr false}
"Mathematica@SplitBy@-@23" {:lang "Mathematica" :algo "SplitBy" :lib "-" :id 23 :doc "http://reference.wolfram.com/language/ref/SplitBy.html" :sig "-" :expr false}
"Python@groupby@itertools@23" {:lang "Python" :algo "groupby" :lib "itertools" :id 23 :doc "https://docs.python.org/3/library/itertools.html#itertools.groupby" :sig "-" :expr false}
"Ruby@chunk_while@Enumerable@23" {:lang "Ruby" :algo "chunk_while" :lib "Enumerable" :id 23 :doc "https://apidock.com/ruby/Enumerable/chunk_while" :sig "-" :expr false}
"Ruby@slice_when@Enumerable@23" {:lang "Ruby" :algo "slice_when" :lib "Enumerable" :id 23 :doc "https://apidock.com/ruby/v2_5_5/Enumerable/slice_when" :sig "-" :expr false}
"Rust@chunk_by@-@23" {:lang "Rust" :algo "chunk_by" :lib "-" :id 23 :doc "https://doc.rust-lang.org/std/primitive.slice.html#method.chunk_by" :sig "-" :expr false}
"TinyAPL@⊆ (partition)@-@23" {:lang "TinyAPL" :algo "⊆ (partition)" :lib "-" :id 23 :doc "https://tinyapl.rubenverg.com/docs/primitive/partition" :sig "-" :expr false}
"Uiua@⊜ (partition)@-@23" {:lang "Uiua" :algo "⊜ (partition)" :lib "-" :id 23 :doc "https://www.uiua.org/docs/partition" :sig "-" :expr false}
"☾@  (groupby)@-@23" {:lang "☾" :algo "  (groupby)" :lib "-" :id 23 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@repeat@Codata.Guarded.Stream@24" {:lang "Agda" :algo "repeat" :lib "Codata.Guarded.Stream" :id 24 :doc "https://agda.github.io/agda-stdlib/v2.2/Codata.Guarded.Stream.html#1216" :sig "-" :expr false}
"C++@repeat@range-v3@24" {:lang "C++" :algo "repeat" :lib "range-v3" :id 24 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-" :expr false}
"Clojure@repeat@core@24" {:lang "Clojure" :algo "repeat" :lib "core" :id 24 :doc "https://clojuredocs.org/clojure.core/repeat" :sig "-" :expr false}
"D@repeat@range@24" {:lang "D" :algo "repeat" :lib "range" :id 24 :doc "https://dlang.org/library/std/range/repeat.html" :sig "-" :expr false}
"Excel@expand@-@24" {:lang "Excel" :algo "expand" :lib "-" :id 24 :doc "https://support.microsoft.com/en-us/office/expand-function-7433fba5-4ad1-41da-a904-d5d95808bc38" :sig "-" :expr false}
"Haskell@repeat@Prelude@24" {:lang "Haskell" :algo "repeat" :lib "Prelude" :id 24 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:repeat" :sig "-" :expr false}
"JavaScript@repeat@ramda@24" {:lang "JavaScript" :algo "repeat" :lib "ramda" :id 24 :doc "https://ramdajs.com/docs/#repeat" :sig "-" :expr false}
"Rust@repeat@iter@24" {:lang "Rust" :algo "repeat" :lib "iter" :id 24 :doc "https://doc.rust-lang.org/std/iter/fn.repeat.html" :sig "-" :expr false}
"Scala@continually@LazyList@24" {:lang "Scala" :algo "continually" :lib "LazyList" :id 24 :doc "https://scala-lang.org/api/current/scala/collection/immutable/LazyList$.html#continually-fffff347" :sig "-" :expr false}
"APL@/ (replicate)@-@25" {:lang "APL" :algo "/ (replicate)" :lib "-" :id 25 :doc "https://aplwiki.com/wiki/Replicate" :sig "-" :expr false}
"Agda@replicate@various@25" {:lang "Agda" :algo "replicate" :lib "various" :id 25 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#5319" :sig "-" :expr false}
"C#@repeat@Enumerable@25" {:lang "C#" :algo "repeat" :lib "Enumerable" :id 25 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.repeat?view=netcore-3.1" :sig "-" :expr false}
"C++@repeat_n@range-v3@25" {:lang "C++" :algo "repeat_n" :lib "range-v3" :id 25 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-" :expr false}
"Haskell@replicate@Prelude@25" {:lang "Haskell" :algo "replicate" :lib "Prelude" :id 25 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:replicate" :sig "-" :expr false}
"J@# (copy)@-@25" {:lang "J" :algo "# (copy)" :lib "-" :id 25 :doc "https://code.jsoftware.com/wiki/Vocabulary/number#dyadic" :sig "-" :expr false}
"JavaScript@repeat@ramda@25" {:lang "JavaScript" :algo "repeat" :lib "ramda" :id 25 :doc "https://ramdajs.com/docs/#repeat" :sig "-" :expr false}
"Julia@repeat@Base@25" {:lang "Julia" :algo "repeat" :lib "Base" :id 25 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.repeat" :sig "-" :expr false}
"Mathematica@ConstantArray@-@25" {:lang "Mathematica" :algo "ConstantArray" :lib "-" :id 25 :doc "http://reference.wolfram.com/language/ref/ConstantArray.html" :sig "-" :expr false}
"Scala@tabulate@various@25" {:lang "Scala" :algo "tabulate" :lib "various" :id 25 :doc "https://scala-lang.org/api/current/scala/Array$.html#tabulate-fffffde9" :sig "-" :expr false}
"TinyAPL@⌿ (replicate)@-@25" {:lang "TinyAPL" :algo "⌿ (replicate)" :lib "-" :id 25 :doc "https://tinyapl.rubenverg.com/docs/primitive/replicate" :sig "-" :expr false}
"Uiua@▽ (keep)@-@25" {:lang "Uiua" :algo "▽ (keep)" :lib "-" :id 25 :doc "https://www.uiua.org/docs/keep" :sig "-" :expr false}
"☾@⛶ⁿ  (enlist)@-@25" {:lang "☾" :algo "⛶ⁿ  (enlist)" :lib "-" :id 25 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@cartesianProductWith@Data.List@26" {:lang "Agda" :algo "cartesianProductWith" :lib "Data.List" :id 26 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#2132" :sig "-" :expr false}
"C++@slide@<ranges>@27" {:lang "C++" :algo "slide" :lib "<ranges>" :id 27 :doc "https://en.cppreference.com/w/cpp/ranges/slide_view" :sig "-" :expr false}
"D@slide@std.range@27" {:lang "D" :algo "slide" :lib "std.range" :id 27 :doc "https://dlang.org/library/std/range/slide.html" :sig "-" :expr false}
"JavaScript@aperture@ramda@27" {:lang "JavaScript" :algo "aperture" :lib "ramda" :id 27 :doc "https://ramdajs.com/docs/#aperture" :sig "-" :expr false}
"Scala@sliding@various@27" {:lang "Scala" :algo "sliding" :lib "various" :id 27 :doc "https://scala-lang.org/api/current/scala/collection/IterableOps.html#sliding-fffffbef" :sig "-" :expr false}
"C++@zip@<ranges>@28" {:lang "C++" :algo "zip" :lib "<ranges>" :id 28 :doc "https://en.cppreference.com/w/cpp/ranges/zip_view" :sig "-" :expr false}
"F#@zip@various@28" {:lang "F#" :algo "zip" :lib "various" :id 28 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-listmodule.html#zip" :sig "-" :expr false}
"F#@zip3@various@28" {:lang "F#" :algo "zip3" :lib "various" :id 28 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-listmodule.html#zip3" :sig "-" :expr false}
"JavaScript@zip@ramda@28" {:lang "JavaScript" :algo "zip" :lib "ramda" :id 28 :doc "https://ramdajs.com/docs/#zip" :sig "-" :expr false}
"Python@zip@-@28" {:lang "Python" :algo "zip" :lib "-" :id 28 :doc "https://docs.python.org/3/library/functions.html#zip" :sig "-" :expr false}
"APL@\\ (expand)@-@29" {:lang "APL" :algo "\\ (expand)" :lib "-" :id 29 :doc "http://microapl.com/apl_help/ch_020_020_860.htm" :sig "-" :expr false}
"Agda@zipWith@various@30" {:lang "Agda" :algo "zipWith" :lib "various" :id 30 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#2734" :sig "-" :expr false}
"C++@transform*@<algorithm>@30" {:lang "C++" :algo "transform*" :lib "<algorithm>" :id 30 :doc "https://en.cppreference.com/w/cpp/algorithm/transform" :sig "-" :expr false}
"Clojure@map*@core@30" {:lang "Clojure" :algo "map*" :lib "core" :id 30 :doc "https://clojuredocs.org/clojure.core/map" :sig "-" :expr false}
"F#@map2@Seq@30" {:lang "F#" :algo "map2" :lib "Seq" :id 30 :doc "https://msdn.microsoft.com/visualfsharpdocs/conceptual/list.map2%5b%27t1%2c%27t2%2c%27u%5d-function-%5bfsharp%5d" :sig "-" :expr false}
"Gleam@map2@list@30" {:lang "Gleam" :algo "map2" :lib "list" :id 30 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#map2" :sig "-" :expr false}
"Haskell@zipWith@Prelude@30" {:lang "Haskell" :algo "zipWith" :lib "Prelude" :id 30 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:zipWith" :sig "-" :expr false}
"JavaScript@map@-@30" {:lang "JavaScript" :algo "map" :lib "-" :id 30 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map" :sig "-" :expr false}
"JavaScript@map@ramda@30" {:lang "JavaScript" :algo "map" :lib "ramda" :id 30 :doc "https://ramdajs.com/docs/#map" :sig "-" :expr false}
"Mathematica@MapThread@-@30" {:lang "Mathematica" :algo "MapThread" :lib "-" :id 30 :doc "http://reference.wolfram.com/language/ref/MapThread.html" :sig "-" :expr false}
"OCaml@map2@List@30" {:lang "OCaml" :algo "map2" :lib "List" :id 30 :doc "https://caml.inria.fr/pub/docs/manual-ocaml/libref/List.html" :sig "-" :expr false}
"Racket@map*@base@30" {:lang "Racket" :algo "map*" :lib "base" :id 30 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Fprivate%2Fmap..rkt%29._map%29%29" :sig "-" :expr false}
"TinyAPL@¨ (each)@-@30" {:lang "TinyAPL" :algo "¨ (each)" :lib "-" :id 30 :doc "https://tinyapl.rubenverg.com/docs/primitive/each" :sig "-" :expr false}
"☾@ꟿ  (starmap)@-@30" {:lang "☾" :algo "ꟿ  (starmap)" :lib "-" :id 30 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Agda@all@various@31" {:lang "Agda" :algo "all" :lib "various" :id 31 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#4910" :sig "-" :expr false}
"C#@All@Linq@31" {:lang "C#" :algo "All" :lib "Linq" :id 31 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.all?view=netcore-3.1" :sig "-" :expr false}
"C++@all_of@<algorithm>@31" {:lang "C++" :algo "all_of" :lib "<algorithm>" :id 31 :doc "https://en.cppreference.com/w/cpp/algorithm/all_any_none_of" :sig "-" :expr false}
"Clojure@every?@core@31" {:lang "Clojure" :algo "every?" :lib "core" :id 31 :doc "https://clojuredocs.org/clojure.core/every_q" :sig "-" :expr false}
"Common Lisp@every@-@31" {:lang "Common Lisp" :algo "every" :lib "-" :id 31 :doc "https://novaspec.org/cl/f_every" :sig "-" :expr false}
"D@all@std.algorithm@31" {:lang "D" :algo "all" :lib "std.algorithm" :id 31 :doc "https://dlang.org/library/std/algorithm/searching/all.all.html" :sig "-" :expr false}
"Elixir@all?@Enum@31" {:lang "Elixir" :algo "all?" :lib "Enum" :id 31 :doc "https://hexdocs.pm/elixir/Enum.html#all?/2" :sig "-" :expr false}
"Elm@all@List@31" {:lang "Elm" :algo "all" :lib "List" :id 31 :doc "https://package.elm-lang.org/packages/elm/core/latest/List#all" :sig "-" :expr false}
"Erlang@all@lists@31" {:lang "Erlang" :algo "all" :lib "lists" :id 31 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#all/2" :sig "-" :expr false}
"Haskell@all@Prelude@31" {:lang "Haskell" :algo "all" :lib "Prelude" :id 31 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/GHC-List.html#v:all" :sig "-" :expr false}
"Java@allMatch@Streams@31" {:lang "Java" :algo "allMatch" :lib "Streams" :id 31 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#allMatch-java.util.function.Predicate-" :sig "-" :expr false}
"JavaScript@every@-@31" {:lang "JavaScript" :algo "every" :lib "-" :id 31 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/every" :sig "-" :expr false}
"JavaScript@all@ramda@31" {:lang "JavaScript" :algo "all" :lib "ramda" :id 31 :doc "https://ramdajs.com/docs/#all" :sig "-" :expr false}
"Julia@all@Base@31" {:lang "Julia" :algo "all" :lib "Base" :id 31 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.all-Tuple%7BAny%7D" :sig "-" :expr false}
"Kotlin@all@sequences@31" {:lang "Kotlin" :algo "all" :lib "sequences" :id 31 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/all.html" :sig "-" :expr false}
"Mathematica@AllTrue@-@31" {:lang "Mathematica" :algo "AllTrue" :lib "-" :id 31 :doc "http://reference.wolfram.com/language/ref/AllTrue.html" :sig "-" :expr false}
"OCaml@for_all@List@31" {:lang "OCaml" :algo "for_all" :lib "List" :id 31 :doc "https://caml.inria.fr/pub/docs/manual-ocaml/libref/List.html" :sig "-" :expr false}
"Python@all@-@31" {:lang "Python" :algo "all" :lib "-" :id 31 :doc "https://docs.python.org/3/library/functions.html#all" :sig "-" :expr false}
"Ruby@all@Enumerable@31" {:lang "Ruby" :algo "all" :lib "Enumerable" :id 31 :doc "https://apidock.com/ruby/Enumerable/all%3F" :sig "-" :expr false}
"Rust@all@iter@31" {:lang "Rust" :algo "all" :lib "iter" :id 31 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.all" :sig "-" :expr false}
"Scala@forall@various@31" {:lang "Scala" :algo "forall" :lib "various" :id 31 :doc "https://scala-lang.org/api/current/scala/collection/IterableOnceOps.html#forall-fffff66f" :sig "-" :expr false}
"Swift@allSatisfy@-@31" {:lang "Swift" :algo "allSatisfy" :lib "-" :id 31 :doc "https://developer.apple.com/documentation/swift/array/2994715-allsatisfy" :sig "-" :expr false}
"Uiua@/× (reduce multiply)@-@31" {:lang "Uiua" :algo "/× (reduce multiply)" :lib "-" :id 31 :doc "https://www.uiua.org/reduce" :sig "-" :expr true}
"☾@⋀  (all)@-@31" {:lang "☾" :algo "⋀  (all)" :lib "-" :id 31 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@≠ (unique mask)@-@32" {:lang "APL" :algo "≠ (unique mask)" :lib "-" :id 32 :doc "https://aplwiki.com/wiki/Nub_Sieve" :sig "-" :expr false}
"BQN@∊ (mark firsts)@-@32" {:lang "BQN" :algo "∊ (mark firsts)" :lib "-" :id 32 :doc "https://mlochbaum.github.io/BQN/doc/selfcmp.html#mark-firsts" :sig "-" :expr false}
"J@~: (nub sieve)@-@32" {:lang "J" :algo "~: (nub sieve)" :lib "-" :id 32 :doc "https://code.jsoftware.com/wiki/Vocabulary/tildeco" :sig "-" :expr false}
"Kap@≠ (unique mask)@-@32" {:lang "Kap" :algo "≠ (unique mask)" :lib "-" :id 32 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_not_equals" :sig "-" :expr false}
"TinyAPL@≠ (nub sieve)@-@32" {:lang "TinyAPL" :algo "≠ (nub sieve)" :lib "-" :id 32 :doc "https://tinyapl.rubenverg.com/docs/primitive/nub_sieve" :sig "-" :expr false}
"Uiua@◰ (unique)@-@32" {:lang "Uiua" :algo "◰ (unique)" :lib "-" :id 32 :doc "https://www.uiua.org/docs/unique" :sig "-" :expr false}
"Common Lisp@butLast@-@33" {:lang "Common Lisp" :algo "butLast" :lib "-" :id 33 :doc "https://novaspec.org/cl/f_butlast" :sig "-" :expr false}
"Haskell@init@Data.List@33" {:lang "Haskell" :algo "init" :lib "Data.List" :id 33 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Prelude.html#v:init" :sig "-" :expr false}
"JavaScript@init@ramda@33" {:lang "JavaScript" :algo "init" :lib "ramda" :id 33 :doc "https://ramdajs.com/docs/#init" :sig "-" :expr false}
"Pharo@allButLast@-@33" {:lang "Pharo" :algo "allButLast" :lib "-" :id 33 :doc "" :sig "-" :expr false}
"C++@drop@<ranges>@34" {:lang "C++" :algo "drop" :lib "<ranges>" :id 34 :doc "https://en.cppreference.com/w/cpp/ranges/drop_view" :sig "-" :expr false}
"Haskell@drop@Prelude@34" {:lang "Haskell" :algo "drop" :lib "Prelude" :id 34 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Prelude.html#v:drop" :sig "-" :expr false}
"JavaScript@drop@ramda@34" {:lang "JavaScript" :algo "drop" :lib "ramda" :id 34 :doc "https://ramdajs.com/docs/#drop" :sig "-" :expr false}
"Julia@drop@Base@34" {:lang "Julia" :algo "drop" :lib "Base" :id 34 :doc "https://docs.julialang.org/en/v1/base/iterators/#Base.Iterators.drop" :sig "-" :expr false}
"Pharo@allButFirst:@Collection@34" {:lang "Pharo" :algo "allButFirst:" :lib "Collection" :id 34 :doc "" :sig "-" :expr false}
"Rust@skip@iter@34" {:lang "Rust" :algo "skip" :lib "iter" :id 34 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.skip" :sig "-" :expr false}
"Scala@drop@various@34" {:lang "Scala" :algo "drop" :lib "various" :id 34 :doc "https://scala-lang.org/api/current/scala/collection/IterableOps.html#drop-aba" :sig "-" :expr false}
"k@_ (drop)@-@34" {:lang "k" :algo "_ (drop)" :lib "-" :id 34 :doc "https://ngn.codeberg.page/k/#" :sig "-" :expr false}
"C++@take@<ranges>@35" {:lang "C++" :algo "take" :lib "<ranges>" :id 35 :doc "https://en.cppreference.com/w/cpp/ranges/take_view" :sig "-" :expr false}
"Haskell@take@Prelude@35" {:lang "Haskell" :algo "take" :lib "Prelude" :id 35 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Prelude.html#v:take" :sig "-" :expr false}
"JavaScript@take@ramda@35" {:lang "JavaScript" :algo "take" :lib "ramda" :id 35 :doc "https://ramdajs.com/docs/#take" :sig "-" :expr false}
"Julia@take@Base@35" {:lang "Julia" :algo "take" :lib "Base" :id 35 :doc "https://docs.julialang.org/en/v1/base/iterators/#Base.Iterators.take" :sig "-" :expr false}
"Pharo@first:@Collection@35" {:lang "Pharo" :algo "first:" :lib "Collection" :id 35 :doc "" :sig "-" :expr false}
"Rust@take@iter@35" {:lang "Rust" :algo "take" :lib "iter" :id 35 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take" :sig "-" :expr false}
"Scala@take@various@35" {:lang "Scala" :algo "take" :lib "various" :id 35 :doc "https://scala-lang.org/api/current/scala/collection/IterableOps.html#take-aba" :sig "-" :expr false}
"k@# (take)@-@35" {:lang "k" :algo "# (take)" :lib "-" :id 35 :doc "https://ngn.codeberg.page/k/#" :sig "-" :expr false}
"C++@minmax_element@<algorithm>@36" {:lang "C++" :algo "minmax_element" :lib "<algorithm>" :id 36 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/minmax_element" :sig "-" :expr false}
"C++@minmax@<algorithm>@36" {:lang "C++" :algo "minmax" :lib "<algorithm>" :id 36 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/minmax" :sig "-" :expr false}
"Julia@extrema@Base@36" {:lang "Julia" :algo "extrema" :lib "Base" :id 36 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.extrema" :sig "-" :expr false}
"Nim@minmax@sequtils@36" {:lang "Nim" :algo "minmax" :lib "sequtils" :id 36 :doc "https://nim-lang.org/docs/sequtils.html#minmax,seq[T]" :sig "-" :expr false}
"Python@minmax@more-itertools@36" {:lang "Python" :algo "minmax" :lib "more-itertools" :id 36 :doc "https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.minmax" :sig "-" :expr false}
"Rust@minmax@itertools@36" {:lang "Rust" :algo "minmax" :lib "itertools" :id 36 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.minmax" :sig "-" :expr false}
"C#@Max@Linq@37" {:lang "C#" :algo "Max" :lib "Linq" :id 37 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.max" :sig "-" :expr false}
"C++@max_element@<algorithm>@37" {:lang "C++" :algo "max_element" :lib "<algorithm>" :id 37 :doc "https://en.cppreference.com/w/cpp/algorithm/max_element" :sig "-" :expr false}
"CUDA@max_element@Thrust@37" {:lang "CUDA" :algo "max_element" :lib "Thrust" :id 37 :doc "https://nvidia.github.io/cccl/thrust/api/function_group__extrema_1gadd477c531dc4a4bf847f70ce73d01464.html" :sig "-" :expr false}
"Common Lisp@max@-@37" {:lang "Common Lisp" :algo "max" :lib "-" :id 37 :doc "https://novaspec.org/cl/f_max" :sig "-" :expr false}
"Elixir@max@Enum@37" {:lang "Elixir" :algo "max" :lib "Enum" :id 37 :doc "https://hexdocs.pm/elixir/Enum.html#max/1" :sig "-" :expr false}
"Erlang@max@lists@31" {:lang "Erlang" :algo "max" :lib "lists" :id 31 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#max/1" :sig "-" :expr false}
"Haskell@maximum@Prelude@37" {:lang "Haskell" :algo "maximum" :lib "Prelude" :id 37 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Prelude.html#v:maximum" :sig "-" :expr false}
"Java@max@Streams@37" {:lang "Java" :algo "max" :lib "Streams" :id 37 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/IntStream.html#max--" :sig "-" :expr false}
"JavaScript@Math.max@-@37" {:lang "JavaScript" :algo "Math.max" :lib "-" :id 37 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/max" :sig "-" :expr false}
"JavaScript@max@ramda@37" {:lang "JavaScript" :algo "max" :lib "ramda" :id 37 :doc "https://ramdajs.com/docs/#max" :sig "-" :expr false}
"Julia@maximum@Base@37" {:lang "Julia" :algo "maximum" :lib "Base" :id 37 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.maximum" :sig "-" :expr false}
"Kotlin@max@Sequence@37" {:lang "Kotlin" :algo "max" :lib "Sequence" :id 37 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/max.html" :sig "-" :expr false}
"OCaml@max@-@37" {:lang "OCaml" :algo "max" :lib "-" :id 37 :doc "https://ocaml.org/api/Stdlib.html#max" :sig "-" :expr false}
"Python@max@-@37" {:lang "Python" :algo "max" :lib "-" :id 37 :doc "https://docs.python.org/3/library/functions.html#max" :sig "-" :expr false}
"Ruby@max@-@37" {:lang "Ruby" :algo "max" :lib "-" :id 37 :doc "https://apidock.com/ruby/Enumerable/max" :sig "-" :expr false}
"Rust@max@iter@37" {:lang "Rust" :algo "max" :lib "iter" :id 37 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.max" :sig "-" :expr false}
"Scala@max@-@37" {:lang "Scala" :algo "max" :lib "-" :id 37 :doc "https://www.scala-lang.org/api/current/scala/collection/IterableOps.html#max-ffffff34" :sig "-" :expr false}
"Swift@max@Sequence@37" {:lang "Swift" :algo "max" :lib "Sequence" :id 37 :doc "https://developer.apple.com/documentation/swift/sequence/max()" :sig "-" :expr false}
"C#@Min@Linq@38" {:lang "C#" :algo "Min" :lib "Linq" :id 38 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.min" :sig "-" :expr false}
"C++@min_element@<algorithm>@38" {:lang "C++" :algo "min_element" :lib "<algorithm>" :id 38 :doc "https://en.cppreference.com/w/cpp/algorithm/min_element" :sig "-" :expr false}
"CUDA@min_element@Thrust@38" {:lang "CUDA" :algo "min_element" :lib "Thrust" :id 38 :doc "https://nvidia.github.io/cccl/thrust/api/function_group__extrema_1ga70a05eafeff24c0553adcbb51d8f052a.html" :sig "-" :expr false}
"Common Lisp@min@-@38" {:lang "Common Lisp" :algo "min" :lib "-" :id 38 :doc "https://novaspec.org/cl/f_max" :sig "-" :expr false}
"Elixir@min@Enum@38" {:lang "Elixir" :algo "min" :lib "Enum" :id 38 :doc "https://hexdocs.pm/elixir/Enum.html#min/1" :sig "-" :expr false}
"Erlang@min@lists@31" {:lang "Erlang" :algo "min" :lib "lists" :id 31 :doc "https://www.erlang.org/doc/apps/stdlib/lists.html#min/1" :sig "-" :expr false}
"Haskell@minimum@Prelude@38" {:lang "Haskell" :algo "minimum" :lib "Prelude" :id 38 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Prelude.html#v:minimum" :sig "-" :expr false}
"Java@min@Streams@38" {:lang "Java" :algo "min" :lib "Streams" :id 38 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/IntStream.html#min--" :sig "-" :expr false}
"JavaScript@Math.min@-@38" {:lang "JavaScript" :algo "Math.min" :lib "-" :id 38 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/min" :sig "-" :expr false}
"JavaScript@min@ramda@38" {:lang "JavaScript" :algo "min" :lib "ramda" :id 38 :doc "https://ramdajs.com/docs/#min" :sig "-" :expr false}
"Julia@minimum@Base@38" {:lang "Julia" :algo "minimum" :lib "Base" :id 38 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.minimum" :sig "-" :expr false}
"Kotlin@min@Sequence@38" {:lang "Kotlin" :algo "min" :lib "Sequence" :id 38 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/min.html" :sig "-" :expr false}
"OCaml@min@-@38" {:lang "OCaml" :algo "min" :lib "-" :id 38 :doc "https://ocaml.org/api/Stdlib.html#min" :sig "-" :expr false}
"Python@min@-@38" {:lang "Python" :algo "min" :lib "-" :id 38 :doc "https://docs.python.org/3/library/functions.html#min" :sig "-" :expr false}
"Ruby@min@-@38" {:lang "Ruby" :algo "min" :lib "-" :id 38 :doc "https://apidock.com/ruby/Enumerable/min" :sig "-" :expr false}
"Rust@min@iter@38" {:lang "Rust" :algo "min" :lib "iter" :id 38 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.min" :sig "-" :expr false}
"Scala@min@-@38" {:lang "Scala" :algo "min" :lib "-" :id 38 :doc "https://www.scala-lang.org/api/current/scala/collection/IterableOps.html#min-ffffff34" :sig "-" :expr false}
"Swift@min@Sequence@38" {:lang "Swift" :algo "min" :lib "Sequence" :id 38 :doc "https://developer.apple.com/documentation/swift/sequence/min()" :sig "-" :expr false}
"APL@⍒ (grade down)@-@39" {:lang "APL" :algo "⍒ (grade down)" :lib "-" :id 39 :doc "https://microapl.com/apl_help/ch_020_020_640.htm" :sig "-" :expr false}
"BQN@⍒ (grade down)@-@39" {:lang "BQN" :algo "⍒ (grade down)" :lib "-" :id 39 :doc "https://mlochbaum.github.io/BQN/help/gradedown_binsdown.html#-%F0%9D%95%A9-grade-down" :sig "-" :expr false}
"APL@⍋ (grade up)@-@40" {:lang "APL" :algo "⍋ (grade up)" :lib "-" :id 40 :doc "https://microapl.com/apl_help/ch_020_020_630.htm" :sig "-" :expr false}
"BQN@⍋ (grade up)@-@40" {:lang "BQN" :algo "⍋ (grade up)" :lib "-" :id 40 :doc "https://mlochbaum.github.io/BQN/help/gradeup_binsup.html#-%F0%9D%95%A9-grade-up" :sig "-" :expr false}
"Agda@_++_@various@42" {:lang "Agda" :algo "_++_" :lib "various" :id 42 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#1734" :sig "-" :expr false}
"C++@concat@<ranges>@42" {:lang "C++" :algo "concat" :lib "<ranges>" :id 42 :doc "https://en.cppreference.com/w/cpp/ranges/concat_view" :sig "-" :expr false}
"Clojure@concat@List@42" {:lang "Clojure" :algo "concat" :lib "List" :id 42 :doc "https://clojuredocs.org/clojure.core/concat" :sig "-" :expr false}
"Elixir@concat@Enum@42" {:lang "Elixir" :algo "concat" :lib "Enum" :id 42 :doc "https://hexdocs.pm/elixir/Enum.html#concat/2" :sig "-" :expr false}
"Elixir@++@Kernel@42" {:lang "Elixir" :algo "++" :lib "Kernel" :id 42 :doc "https://hexdocs.pm/elixir/Kernel.html#++/2" :sig "-" :expr false}
"Go@append@-@42" {:lang "Go" :algo "append" :lib "-" :id 42 :doc "https://pkg.go.dev/builtin#append" :sig "-" :expr false}
"Haskell@++@Prelude@42" {:lang "Haskell" :algo "++" :lib "Prelude" :id 42 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:-43--43-" :sig "-" :expr false}
"JavaScript@concat@-@42" {:lang "JavaScript" :algo "concat" :lib "-" :id 42 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/concat" :sig "-" :expr false}
"JavaScript@concat@ramda@42" {:lang "JavaScript" :algo "concat" :lib "ramda" :id 42 :doc "https://ramdajs.com/docs/#concat" :sig "-" :expr false}
"Julia@vcat@Base@42" {:lang "Julia" :algo "vcat" :lib "Base" :id 42 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.vcat" :sig "-" :expr false}
"Julia@append!@Base@42" {:lang "Julia" :algo "append!" :lib "Base" :id 42 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.append!" :sig "-" :expr false}
"Mathematica@Append@-@42" {:lang "Mathematica" :algo "Append" :lib "-" :id 42 :doc "http://reference.wolfram.com/language/ref/Append.html" :sig "-" :expr false}
"Mathematica@AppendTo@-@42" {:lang "Mathematica" :algo "AppendTo" :lib "-" :id 42 :doc "http://reference.wolfram.com/language/ref/AppendTo.html" :sig "-" :expr false}
"Python@append@list@42" {:lang "Python" :algo "append" :lib "list" :id 42 :doc "https://docs.python.org/3/tutorial/introduction.html#lists" :sig "-" :expr false}
"Python@chain@itertools@42" {:lang "Python" :algo "chain" :lib "itertools" :id 42 :doc "https://docs.python.org/3/library/itertools.html#itertools.chain" :sig "-" :expr false}
"Racket@append@base@42" {:lang "Racket" :algo "append" :lib "base" :id 42 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28quote._~23~25kernel%29._append%29%29" :sig "-" :expr false}
"Rust@chain@iter@42" {:lang "Rust" :algo "chain" :lib "iter" :id 42 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.chain" :sig "-" :expr false}
"Rust@append@vec@42" {:lang "Rust" :algo "append" :lib "vec" :id 42 :doc "https://doc.rust-lang.org/std/vec/struct.Vec.html#method.append" :sig "-" :expr false}
"Scala@appendedAll@various@42" {:lang "Scala" :algo "appendedAll" :lib "various" :id 42 :doc "https://scala-lang.org/api/current/scala/collection/immutable/IndexedSeqOps.html#appendedAll-1dd" :sig "-" :expr false}
"Scala@++@various@42" {:lang "Scala" :algo "++" :lib "various" :id 42 :doc "https://scala-lang.org/api/current/scala/collection/ArrayOps.html#++-fffff738" :sig "-" :expr false}
"Swift@+@Array@42" {:lang "Swift" :algo "+" :lib "Array" :id 42 :doc "https://developer.apple.com/documentation/swift/array/2963457" :sig "-" :expr false}
"TinyAPL@⍪ (catenate)@-@42" {:lang "TinyAPL" :algo "⍪ (catenate)" :lib "-" :id 42 :doc "https://tinyapl.rubenverg.com/docs/primitive/catenate" :sig "-" :expr false}
"Uiua@⊂ (join)@-@42" {:lang "Uiua" :algo "⊂ (join)" :lib "-" :id 42 :doc "https://www.uiua.org/docs/join" :sig "-" :expr false}
"☾@+  (plus)@-@42" {:lang "☾" :algo "+  (plus)" :lib "-" :id 42 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"☾@󰌷  (chain)@-@42" {:lang "☾" :algo "󰌷  (chain)" :lib "-" :id 42 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"Clojure@apply =@-@51" {:lang "Clojure" :algo "apply =" :lib "-" :id 51 :doc "" :sig "-" :expr false}
"Common Lisp@apply@-@51" {:lang "Common Lisp" :algo "apply" :lib "-" :id 51 :doc "https://novaspec.org/cl/f_apply" :sig "-" :expr false}
"Haskell@allEqual@Data.List.HT@51" {:lang "Haskell" :algo "allEqual" :lib "Data.List.HT" :id 51 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:allEqual" :sig "-" :expr false}
"JavaScript@allEqual@bbo@51" {:lang "JavaScript" :algo "allEqual" :lib "bbo" :id 51 :doc "https://tnfe.github.io/bbo/#allequal" :sig "-" :expr false}
"Python@all_equal@more-itertools@51" {:lang "Python" :algo "all_equal" :lib "more-itertools" :id 51 :doc "https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.all_equal" :sig "-" :expr false}
"Racket@apply =@-@51" {:lang "Racket" :algo "apply =" :lib "-" :id 51 :doc "" :sig "-" :expr false}
"Uiua@/×⧈≍@-@51" {:lang "Uiua" :algo "/×⧈≍" :lib "-" :id 51 :doc "" :sig "-" :expr true}
"APL@,∘.,@-@100" {:lang "APL" :algo ",∘.," :lib "-" :id 100 :doc "" :sig "-" :expr true}
"Agda@cartesianProduct@Data.List@100" {:lang "Agda" :algo "cartesianProduct" :lib "Data.List" :id 100 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#2317" :sig "-" :expr false}
"BQN@⥊⋈⌜@-@100" {:lang "BQN" :algo "⥊⋈⌜" :lib "-" :id 100 :doc "" :sig "-" :expr true}
"C++@cartesian_product@range-v3@100" {:lang "C++" :algo "cartesian_product" :lib "range-v3" :id 100 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-" :expr false}
"C++@cartesian_product@<ranges>@100" {:lang "C++" :algo "cartesian_product" :lib "<ranges>" :id 100 :doc "https://en.cppreference.com/w/cpp/ranges/cartesian_product_view" :sig "-" :expr false}
"D@cartesianProduct@std.algorithm.setops@100" {:lang "D" :algo "cartesianProduct" :lib "std.algorithm.setops" :id 100 :doc "https://dlang.org/library/std/algorithm/setops/cartesian_product.html" :sig "-" :expr false}
"F#@allPairs@List@100" {:lang "F#" :algo "allPairs" :lib "List" :id 100 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-listmodule.html#allPairs" :sig "-" :expr false}
"JavaScript@xprod@ramda@100" {:lang "JavaScript" :algo "xprod" :lib "ramda" :id 100 :doc "https://ramdajs.com/docs/#xprod" :sig "-" :expr false}
"Julia@product@IterTools@100" {:lang "Julia" :algo "product" :lib "IterTools" :id 100 :doc "https://juliacollections.github.io/IterTools.jl/v0.2.1/#product(xs...)-1" :sig "-" :expr false}
"Kap@,,⌻@-@100" {:lang "Kap" :algo ",,⌻" :lib "-" :id 100 :doc "" :sig "-" :expr true}
"Python@product@itertools@100" {:lang "Python" :algo "product" :lib "itertools" :id 100 :doc "https://docs.python.org/3/library/itertools.html#itertools.product" :sig "-" :expr false}
"Racket@cartesian-product@base@100" {:lang "Racket" :algo "cartesian-product" :lib "base" :id 100 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._cartesian-product%29%29" :sig "-" :expr false}
"Ruby@product@Array@100" {:lang "Ruby" :algo "product" :lib "Array" :id 100 :doc "https://apidock.com/ruby/Array/product" :sig "-" :expr false}
"Rust@cartesian_product@itertools@100" {:lang "Rust" :algo "cartesian_product" :lib "itertools" :id 100 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.cartesian_product" :sig "-" :expr false}
"SQL@CROSS JOIN@-@100" {:lang "SQL" :algo "CROSS JOIN" :lib "-" :id 100 :doc "https://www.w3resource.com/sql/joins/cross-join.php" :sig "-" :expr false}
"q@cross@-@100" {:lang "q" :algo "cross" :lib "-" :id 100 :doc "https://code.kx.com/q4m3/A_Built-in_Functions/#a18-cross" :sig "-" :expr false}
"☾@⨉  (cross product)@-@100" {:lang "☾" :algo "⨉  (cross product)" :lib "-" :id 100 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⌽ (rotate)@-@4104" {:lang "APL" :algo "⌽ (rotate)" :lib "-" :id 4104 :doc "https://microapl.com/apl_help/ch_020_020_520.htm" :sig "-" :expr false}
"BQN@⌽ (rotate)@-@4104" {:lang "BQN" :algo "⌽ (rotate)" :lib "-" :id 4104 :doc "https://mlochbaum.github.io/BQN/doc/reverse.html" :sig "-" :expr false}
"C++@rotate@<algorithm>@4104" {:lang "C++" :algo "rotate" :lib "<algorithm>" :id 4104 :doc "https://en.cppreference.com/w/cpp/algorithm/rotate" :sig "-" :expr false}
"D@bringToFront@algorithm.mutation@4104" {:lang "D" :algo "bringToFront" :lib "algorithm.mutation" :id 4104 :doc "https://dlang.org/library/std/algorithm/mutation/bring_to_front.html" :sig "-" :expr false}
"F#@permute@various@4104" {:lang "F#" :algo "permute" :lib "various" :id 4104 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-arraymodule.html#permute" :sig "-" :expr false}
"J@|. (rotate)@-@4104" {:lang "J" :algo "|. (rotate)" :lib "-" :id 4104 :doc "https://code.jsoftware.com/wiki/Vocabulary/bardot#dyadic" :sig "-" :expr false}
"Java@rotate@Collections@4104" {:lang "Java" :algo "rotate" :lib "Collections" :id 4104 :doc "https://docs.oracle.com/javase/7/docs/api/java/util/Collections.html#rotate(java.util.List,%20int)" :sig "-" :expr false}
"Julia@circshift@-@4104" {:lang "Julia" :algo "circshift" :lib "-" :id 4104 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.circshift" :sig "-" :expr false}
"Mathematica@RotateLeft@-@4104" {:lang "Mathematica" :algo "RotateLeft" :lib "-" :id 4104 :doc "http://reference.wolfram.com/language/ref/RotateLeft.html" :sig "-" :expr false}
"Mathematica@RotateRight@-@4104" {:lang "Mathematica" :algo "RotateRight" :lib "-" :id 4104 :doc "http://reference.wolfram.com/language/ref/RotateRight.html" :sig "-" :expr false}
"Python@rotate@collections@4104" {:lang "Python" :algo "rotate" :lib "collections" :id 4104 :doc "https://docs.python.org/3/library/collections.html#collections.deque.rotate" :sig "-" :expr false}
"Ruby@rotate@Array@4104" {:lang "Ruby" :algo "rotate" :lib "Array" :id 4104 :doc "https://apidock.com/ruby/v2_5_5/Array/rotate" :sig "-" :expr false}
"Rust@rotate_left@slice@4104" {:lang "Rust" :algo "rotate_left" :lib "slice" :id 4104 :doc "https://doc.rust-lang.org/stable/std/primitive.slice.html#method.rotate_left" :sig "-" :expr false}
"TinyAPL@⊖ (rotate)@-@4104" {:lang "TinyAPL" :algo "⊖ (rotate)" :lib "-" :id 4104 :doc "https://tinyapl.rubenverg.com/docs/primitive/rotate" :sig "-" :expr false}
"Uiua@↻ (rotate)@-@4104" {:lang "Uiua" :algo "↻ (rotate)" :lib "-" :id 4104 :doc "https://www.uiua.org/docs/rotate" :sig "-" :expr false}
"q@rotate@Core@4104" {:lang "q" :algo "rotate" :lib "Core" :id 4104 :doc "https://code.kx.com/q4m3/A_Built-in_Functions/#a79-rotate" :sig "-" :expr false}
"☾@⪡  (rotate left)@-@4104" {:lang "☾" :algo "⪡  (rotate left)" :lib "-" :id 4104 :doc "https://ganer.xyz/moon/?code=%22docs%20are%20WIP%22%E2%A8%9D%F3%B0%BB%BE%E2%A8%81%E2%98%BE" :sig "-" :expr false}
"APL@⌽ (reverse)@-@4321" {:lang "APL" :algo "⌽ (reverse)" :lib "-" :id 4321 :doc "https://help.dyalog.com/17.1/Content/Language/Primitive%20Functions/Reverse.htm" :sig "-" :expr false}
"BQN@⌽ (reverse)@-@4321" {:lang "BQN" :algo "⌽ (reverse)" :lib "-" :id 4321 :doc "https://mlochbaum.github.io/BQN/doc/reverse.html" :sig "-" :expr false}
"C#@Reverse@Enumerable@4321" {:lang "C#" :algo "Reverse" :lib "Enumerable" :id 4321 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.reverse" :sig "-" :expr false}
"C++@reverse@<ranges>@4321" {:lang "C++" :algo "reverse" :lib "<ranges>" :id 4321 :doc "https://en.cppreference.com/w/cpp/algorithm/ranges/reverse" :sig "-" :expr false}
"Go@Reverse@slices@4321" {:lang "Go" :algo "Reverse" :lib "slices" :id 4321 :doc "https://pkg.go.dev/slices#Reverse" :sig "-" :expr false}
"J@|. (reverse)@-@4321" {:lang "J" :algo "|. (reverse)" :lib "-" :id 4321 :doc "https://code.jsoftware.com/wiki/Vocabulary/bardot" :sig "-" :expr false}
"JavaScript@reverse@-@4321" {:lang "JavaScript" :algo "reverse" :lib "-" :id 4321 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reverse" :sig "-" :expr false}
"JavaScript@reverse@ramda@4321" {:lang "JavaScript" :algo "reverse" :lib "ramda" :id 4321 :doc "https://ramdajs.com/docs/#reverse" :sig "-" :expr true}
"Julia@reverse@Base@4321" {:lang "Julia" :algo "reverse" :lib "Base" :id 4321 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.reverse-Tuple%7BAbstractVector%7D" :sig "-" :expr false}
"Kap@⌽ (reverse)@-@4321" {:lang "Kap" :algo "⌽ (reverse)" :lib "-" :id 4321 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_rotate_horizontallyreverse_horizontally" :sig "-" :expr false}
"Python@reversed@-@4321" {:lang "Python" :algo "reversed" :lib "-" :id 4321 :doc "https://docs.python.org/3/library/functions.html#reversed" :sig "-" :expr false}
"Python@reverse@-@4321" {:lang "Python" :algo "reverse" :lib "-" :id 4321 :doc "https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types" :sig "-" :expr false}
"Rust@rev@iter@4321" {:lang "Rust" :algo "rev" :lib "iter" :id 4321 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.rev" :sig "-" :expr false}
"TinyAPL@⊖ (reverse)@-@4321" {:lang "TinyAPL" :algo "⊖ (reverse)" :lib "-" :id 4321 :doc "https://tinyapl.rubenverg.com/docs/primitive/reverse" :sig "-" :expr false}
"Uiua@⇌ (reverse)@-@4321" {:lang "Uiua" :algo "⇌ (reverse)" :lib "-" :id 4321 :doc "https://www.uiua.org/docs/reverse" :sig "-" :expr false}
"APL@∘ (compose)@-@5020" {:lang "APL" :algo "∘ (compose)" :lib "-" :id 5020 :doc "https://help.dyalog.com/latest/#Language/Primitive%20Operators/Beside.htm" :sig "-" :expr false}
"BQN@∘ (atop)@-@5020" {:lang "BQN" :algo "∘ (atop)" :lib "-" :id 5020 :doc "https://mlochbaum.github.io/BQN/doc/compose.html" :sig "-" :expr false}
"J@@: (at)@-@5020" {:lang "J" :algo "@: (at)" :lib "-" :id 5020 :doc "https://code.jsoftware.com/wiki/Vocabulary/atco" :sig "-" :expr false}
"JavaScript@pipe@ramda@5020" {:lang "JavaScript" :algo "pipe" :lib "ramda" :id 5020 :doc "https://ramdajs.com/docs/#pipe" :sig "-" :expr false}
"TinyAPL@∘ (after)@-@5020" {:lang "TinyAPL" :algo "∘ (after)" :lib "-" :id 5020 :doc "https://tinyapl.rubenverg.com/docs/primitive/after" :sig "-" :expr false}
"APL@⍤ (atop)@-@5021" {:lang "APL" :algo "⍤ (atop)" :lib "-" :id 5021 :doc "https://help.dyalog.com/latest/#Language/Primitive%20Operators/Atop.htm" :sig "-" :expr false}
"BQN@∘ (atop)@-@5021" {:lang "BQN" :algo "∘ (atop)" :lib "-" :id 5021 :doc "https://mlochbaum.github.io/BQN/doc/compose.html" :sig "-" :expr false}
"J@@: (at)@-@5021" {:lang "J" :algo "@: (at)" :lib "-" :id 5021 :doc "https://code.jsoftware.com/wiki/Vocabulary/atco" :sig "-" :expr false}
"TinyAPL@⍤ (atop)@-@5021" {:lang "TinyAPL" :algo "⍤ (atop)" :lib "-" :id 5021 :doc "https://tinyapl.rubenverg.com/docs/primitive/atop" :sig "-" :expr false}
"APL@⍨ (commute)@-@5030" {:lang "APL" :algo "⍨ (commute)" :lib "-" :id 5030 :doc "https://help.dyalog.com/latest/#Language/Primitive%20Operators/Commute.htm" :sig "-" :expr false}
"BQN@˜ (swap)@-@5030" {:lang "BQN" :algo "˜ (swap)" :lib "-" :id 5030 :doc "https://mlochbaum.github.io/BQN/doc/swap.html" :sig "-" :expr false}
"C++@flip@boost::hana@5030" {:lang "C++" :algo "flip" :lib "boost::hana" :id 5030 :doc "https://www.boost.org/doc/libs/1_88_0/boost/hana/functional/flip.hpp" :sig "-" :expr false}
"Haskell@flip@-@5030" {:lang "Haskell" :algo "flip" :lib "-" :id 5030 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Prelude.html#v:flip" :sig "-" :expr false}
"J@~ (passive)@-@5030" {:lang "J" :algo "~ (passive)" :lib "-" :id 5030 :doc "https://code.jsoftware.com/wiki/Vocabulary/tilde#dyadic" :sig "-" :expr false}
"JavaScript@flip@ramda@5030" {:lang "JavaScript" :algo "flip" :lib "ramda" :id 5030 :doc "https://ramdajs.com/docs/#flip" :sig "-" :expr false}
"Kap@⍨ (commute)@-@5030" {:lang "Kap" :algo "⍨ (commute)" :lib "-" :id 5030 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_commuteduplicate" :sig "-" :expr false}
"TinyAPL@⍨ (commute)@-@5030" {:lang "TinyAPL" :algo "⍨ (commute)" :lib "-" :id 5030 :doc "https://tinyapl.rubenverg.com/docs/primitive/commute" :sig "-" :expr false}
"Uiua@: (flip)@-@5030" {:lang "Uiua" :algo ": (flip)" :lib "-" :id 5030 :doc "https://www.uiua.org/docs/flip" :sig "-" :expr false}
"Uiua@˜ (backward)@-@5030" {:lang "Uiua" :algo "˜ (backward)" :lib "-" :id 5030 :doc "https://www.uiua.org/docs/backward" :sig "-" :expr false}
"APL@∘ (beside)@-@5040" {:lang "APL" :algo "∘ (beside)" :lib "-" :id 5040 :doc "https://help.dyalog.com/latest/#Language/Primitive%20Operators/Beside.htm" :sig "-" :expr false}
"BQN@⟜ (after)@-@5040" {:lang "BQN" :algo "⟜ (after)" :lib "-" :id 5040 :doc "https://mlochbaum.github.io/BQN/doc/hook.html" :sig "-" :expr false}
"Kap@∘ (compose)@-@5040" {:lang "Kap" :algo "∘ (compose)" :lib "-" :id 5040 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_compose" :sig "-" :expr false}
"TinyAPL@⟜ (right hook)@-@5040" {:lang "TinyAPL" :algo "⟜ (right hook)" :lib "-" :id 5040 :doc "https://tinyapl.rubenverg.com/docs/primitive/right_hook" :sig "-" :expr false}
"Uiua@⊙ (dip)@-@5040" {:lang "Uiua" :algo "⊙ (dip)" :lib "-" :id 5040 :doc "https://www.uiua.org/docs/dip" :sig "-" :expr false}
"APL@⊢ (same)@-@5090" {:lang "APL" :algo "⊢ (same)" :lib "-" :id 5090 :doc "https://help.dyalog.com/17.1/Content/Language/Primitive%20Functions/Same.htm" :sig "-" :expr false}
"BQN@⊢ (right)@-@5090" {:lang "BQN" :algo "⊢ (right)" :lib "-" :id 5090 :doc "https://mlochbaum.github.io/BQN/doc/identity.html" :sig "-" :expr false}
"C++@identity@<functional>@5090" {:lang "C++" :algo "identity" :lib "<functional>" :id 5090 :doc "https://en.cppreference.com/w/cpp/utility/functional/identity" :sig "-" :expr false}
"J@] (same)@-@5090" {:lang "J" :algo "] (same)" :lib "-" :id 5090 :doc "https://code.jsoftware.com/wiki/Vocabulary/squarert" :sig "-" :expr false}
"JavaScript@identity@ramda@5090" {:lang "JavaScript" :algo "identity" :lib "ramda" :id 5090 :doc "https://ramdajs.com/docs/#identity" :sig "-" :expr false}
"Kap@⊢ (identity)@-@5090" {:lang "Kap" :algo "⊢ (identity)" :lib "-" :id 5090 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_identityright" :sig "-" :expr false}
"TinyAPL@⊢ (same)@-@5090" {:lang "TinyAPL" :algo "⊢ (same)" :lib "-" :id 5090 :doc "https://tinyapl.rubenverg.com/docs/primitive/right" :sig "-" :expr false}
"Uiua@∘ (identity)@-@5090" {:lang "Uiua" :algo "∘ (identity)" :lib "-" :id 5090 :doc "https://www.uiua.org/docs/identity" :sig "-" :expr false}
"APL@⊣ (left)@-@5110" {:lang "APL" :algo "⊣ (left)" :lib "-" :id 5110 :doc "https://help.dyalog.com/17.1/Content/Language/Primitive%20Functions/Left.htm" :sig "-" :expr false}
"BQN@⊣ (left)@-@5110" {:lang "BQN" :algo "⊣ (left)" :lib "-" :id 5110 :doc "https://mlochbaum.github.io/BQN/doc/identity.html" :sig "-" :expr false}
"J@[ (left)@-@5110" {:lang "J" :algo "[ (left)" :lib "-" :id 5110 :doc "https://code.jsoftware.com/wiki/Vocabulary/squarelt" :sig "-" :expr false}
"Kap@⊣ (left)@-@5110" {:lang "Kap" :algo "⊣ (left)" :lib "-" :id 5110 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_hideleft" :sig "-" :expr false}
"TinyAPL@⊣ (left)@-@5110" {:lang "TinyAPL" :algo "⊣ (left)" :lib "-" :id 5110 :doc "https://tinyapl.rubenverg.com/docs/primitive/left" :sig "-" :expr false}
"Uiua@⊙◌ (dip pop)@-@5110" {:lang "Uiua" :algo "⊙◌ (dip pop)" :lib "-" :id 5110 :doc "https://www.uiua.org/tutorial/morestack#planet-notation" :sig "-" :expr true}
"APL@⊢ (right)@-@5111" {:lang "APL" :algo "⊢ (right)" :lib "-" :id 5111 :doc "https://help.dyalog.com/17.1/Content/Language/Primitive%20Functions/Right.htm" :sig "-" :expr false}
"BQN@⊢ (right)@-@5111" {:lang "BQN" :algo "⊢ (right)" :lib "-" :id 5111 :doc "https://mlochbaum.github.io/BQN/doc/identity.html" :sig "-" :expr false}
"J@] (right)@-@5111" {:lang "J" :algo "] (right)" :lib "-" :id 5111 :doc "https://code.jsoftware.com/wiki/Vocabulary/squarert#dyadic" :sig "-" :expr false}
"Kap@⊢ (right)@-@5111" {:lang "Kap" :algo "⊢ (right)" :lib "-" :id 5111 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_identityright" :sig "-" :expr false}
"TinyAPL@⊢ (right)@-@5111" {:lang "TinyAPL" :algo "⊢ (right)" :lib "-" :id 5111 :doc "https://tinyapl.rubenverg.com/docs/primitive/right" :sig "-" :expr false}
"Uiua@⋅∘ (gap identity)@-@5111" {:lang "Uiua" :algo "⋅∘ (gap identity)" :lib "-" :id 5111 :doc "https://www.uiua.org/tutorial/morestack#planet-notation" :sig "-" :expr true}
"BQN@⟜ (after)@-@5190" {:lang "BQN" :algo "⟜ (after)" :lib "-" :id 5190 :doc "https://mlochbaum.github.io/BQN/doc/hook.html" :sig "-" :expr false}
"Kap@∘ (compose)@-@5190" {:lang "Kap" :algo "∘ (compose)" :lib "-" :id 5190 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_compose" :sig "-" :expr false}
"TinyAPL@⟜ (right hook)@-@5190" {:lang "TinyAPL" :algo "⟜ (right hook)" :lib "-" :id 5190 :doc "https://tinyapl.rubenverg.com/docs/primitive/right_hook" :sig "-" :expr false}
"Uiua@⟜ (on)@-@5190" {:lang "Uiua" :algo "⟜ (on)" :lib "-" :id 5190 :doc "https://www.uiua.org/docs/on" :sig "-" :expr false}
"APL@⍨ (duplicate)@-@5230" {:lang "APL" :algo "⍨ (duplicate)" :lib "-" :id 5230 :doc "https://help.dyalog.com/latest/index.htm#Language/Primitive%20Operators/Commute.htm" :sig "-" :expr false}
"BQN@˜ (self)@-@5230" {:lang "BQN" :algo "˜ (self)" :lib "-" :id 5230 :doc "https://mlochbaum.github.io/BQN/doc/swap.html" :sig "-" :expr false}
"Haskell@join@Control.Monad@5230" {:lang "Haskell" :algo "join" :lib "Control.Monad" :id 5230 :doc "https://hackage.haskell.org/package/base-4.21.0.0/docs/Control-Monad.html#v:join" :sig "-" :expr false}
"J@~ (reflex)@-@5230" {:lang "J" :algo "~ (reflex)" :lib "-" :id 5230 :doc "https://code.jsoftware.com/wiki/Vocabulary/tilde" :sig "-" :expr false}
"Kap@⍨ (duplicate)@-@5230" {:lang "Kap" :algo "⍨ (duplicate)" :lib "-" :id 5230 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_commuteduplicate" :sig "-" :expr false}
"TinyAPL@⍨ (duplicate)@-@5230" {:lang "TinyAPL" :algo "⍨ (duplicate)" :lib "-" :id 5230 :doc "https://tinyapl.rubenverg.com/docs/primitive/duplicate" :sig "-" :expr false}
"Uiua@. (duplicate)@-@5230" {:lang "Uiua" :algo ". (duplicate)" :lib "-" :id 5230 :doc "https://www.uiua.org/docs/duplicate" :sig "-" :expr false}
"BQN@⊸ (before)@-@5540" {:lang "BQN" :algo "⊸ (before)" :lib "-" :id 5540 :doc "https://mlochbaum.github.io/BQN/doc/hook.html" :sig "-" :expr false}
"Kap@⍛ (inverse compose)@-@5540" {:lang "Kap" :algo "⍛ (inverse compose)" :lib "-" :id 5540 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_inverse_compose" :sig "-" :expr false}
"TinyAPL@⊸ (left hook)@-@5540" {:lang "TinyAPL" :algo "⊸ (left hook)" :lib "-" :id 5540 :doc "https://tinyapl.rubenverg.com/docs/primitive/left_hook" :sig "-" :expr false}
"BQN@⊸ (before)@-@5690" {:lang "BQN" :algo "⊸ (before)" :lib "-" :id 5690 :doc "https://mlochbaum.github.io/BQN/doc/hook.html" :sig "-" :expr false}
"Kap@⍛ (inverse compose)@-@5690" {:lang "Kap" :algo "⍛ (inverse compose)" :lib "-" :id 5690 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_inverse_compose" :sig "-" :expr false}
"TinyAPL@⊸ (left hook)@-@5690" {:lang "TinyAPL" :algo "⊸ (left hook)" :lib "-" :id 5690 :doc "https://tinyapl.rubenverg.com/docs/primitive/left_hook" :sig "-" :expr false}
"Uiua@⊸ (by)@-@5690" {:lang "Uiua" :algo "⊸ (by)" :lib "-" :id 5690 :doc "https://www.uiua.org/docs/by" :sig "-" :expr false}
"Kap@«» (fork)@-@5720" {:lang "Kap" :algo "«» (fork)" :lib "-" :id 5720 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_and_fork" :sig "-" :expr false}
"TinyAPL@«» (fork)@-@5720" {:lang "TinyAPL" :algo "«» (fork)" :lib "-" :id 5720 :doc "https://tinyapl.rubenverg.com/docs/primitive/fork" :sig "-" :expr false}
"Uiua@⊃ (fork)@-@5720" {:lang "Uiua" :algo "⊃ (fork)" :lib "-" :id 5720 :doc "https://www.uiua.org/docs/fork" :sig "-" :expr false}
"APL@⍥ (over)@-@5740" {:lang "APL" :algo "⍥ (over)" :lib "-" :id 5740 :doc "https://help.dyalog.com/latest/#Language/Primitive%20Operators/Over.htm" :sig "-" :expr false}
"BQN@○ (over)@-@5740" {:lang "BQN" :algo "○ (over)" :lib "-" :id 5740 :doc "https://mlochbaum.github.io/BQN/doc/compose.html" :sig "-" :expr false}
"J@&: (appose)@-@5740" {:lang "J" :algo "&: (appose)" :lib "-" :id 5740 :doc "https://code.jsoftware.com/wiki/Vocabulary/ampco" :sig "-" :expr false}
"Kap@⍥ (over)@-@5740" {:lang "Kap" :algo "⍥ (over)" :lib "-" :id 5740 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_over" :sig "-" :expr false}
"TinyAPL@⍥ (over)@-@5740" {:lang "TinyAPL" :algo "⍥ (over)" :lib "-" :id 5740 :doc "https://tinyapl.rubenverg.com/docs/primitive/over" :sig "-" :expr false}
"Uiua@∩ (both)@-@5740" {:lang "Uiua" :algo "∩ (both)" :lib "-" :id 5740 :doc "https://www.uiua.org/docs/both" :sig "-" :expr false}
"Uiua@⍥ (repeat)@-@5760" {:lang "Uiua" :algo "⍥ (repeat)" :lib "-" :id 5760 :doc "https://www.uiua.org/docs/repeat" :sig "-" :expr false}
"BQN@⍟ (repeat)@-@5760" {:lang "BQN" :algo "⍟ (repeat)" :lib "-" :id 5760 :doc "https://mlochbaum.github.io/BQN/doc/repeat.html" :sig "-" :expr false}
"APL@⍣ (power)@-@5760" {:lang "APL" :algo "⍣ (power)" :lib "-" :id 5760 :doc "https://aplwiki.com/wiki/Power_(operator)" :sig "-" :expr false}
"Uiua@⧈-@-@41" {:lang "Uiua" :algo "⧈-" :lib "-" :id 41 :doc "" :sig "-" :expr true}
"APL@2-/@-@41" {:lang "APL" :algo "2-/" :lib "-" :id 41 :doc "" :sig "-" :expr true}
"Kap@2-/@-@41" {:lang "Kap" :algo "2-/" :lib "-" :id 41 :doc "" :sig "-" :expr true}
"BQN@-´˘2↕@-@41" {:lang "BQN" :algo "-´˘2↕" :lib "-" :id 41 :doc "" :sig "-" :expr true}
"Julia@diff@Base@41" {:lang "Julia" :algo "diff" :lib "Base" :id 41 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.diff" :sig "-" :expr false}
"MATLAB@diff@-@41" {:lang "MATLAB" :algo "diff" :lib "-" :id 41 :doc "https://www.mathworks.com/help/matlab/ref/diff.html" :sig "-" :expr false}
"Python@diff@NumPy@41" {:lang "Python" :algo "diff" :lib "NumPy" :id 41 :doc "https://numpy.org/doc/stable/reference/generated/numpy.diff.html" :sig "-" :expr false}
"R@diff@base@41" {:lang "R" :algo "diff" :lib "base" :id 41 :doc "https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/diff" :sig "-" :expr false}
"q@deltas@-@41" {:lang "q" :algo "deltas" :lib "-" :id 41 :doc "https://code.kx.com/q/ref/deltas/" :sig "-" :expr false}
"APL@⌈/@-@37" {:lang "APL" :algo "⌈/" :lib "-" :id 37 :doc "" :sig "-" :expr true}
"BQN@⌈´@-@37" {:lang "BQN" :algo "⌈´" :lib "-" :id 37 :doc "" :sig "-" :expr true}
"Kap@⌈/@-@37" {:lang "Kap" :algo "⌈/" :lib "-" :id 37 :doc "" :sig "-" :expr true}
"Uiua@/↥@-@37" {:lang "Uiua" :algo "/↥" :lib "-" :id 37 :doc "" :sig "-" :expr true}
"Rust@all_equal@itertools@51" {:lang "Rust" :algo "all_equal" :lib "itertools" :id 51 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.all_equal" :sig "-" :expr false}
"C++@zip_transform@<ranges>@30" {:lang "C++" :algo "zip_transform" :lib "<ranges>" :id 30 :doc "https://en.cppreference.com/w/cpp/ranges/zip_transform_view.html" :sig "-" :expr false}
})
