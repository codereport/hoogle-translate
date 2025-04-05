(ns ht.data)

(def by-key-map
  {
"J@/ (insert)@-" {:lang "J" :algo "/ (insert)" :lib "-" :id 1 :doc "https://code.jsoftware.com/wiki/Vocabulary/slash" :sig "-"}
"APL@/ (reduce)@-" {:lang "APL" :algo "/ (reduce)" :lib "-" :id 1 :doc "http://microapl.com/apl_help/ch_020_020_800.htm" :sig "-"}
"C++@accumulate@<numeric>" {:lang "C++" :algo "accumulate" :lib "<numeric>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/accumulate" :sig "-"}
"C#@Aggregate@Enumerable" {:lang "C#" :algo "Aggregate" :lib "Enumerable" :id 1 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.aggregate" :sig "-"}
"Rust@fold@trait.Iterator" {:lang "Rust" :algo "fold" :lib "trait.Iterator" :id 1 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.fold" :sig "-"}
"Haskell@foldl@Data.List" {:lang "Haskell" :algo "foldl" :lib "Data.List" :id 1 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:foldl" :sig "-"}
"Racket@foldl@base" {:lang "Racket" :algo "foldl" :lib "base" :id 1 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Fprivate%2Flist..rkt%29._foldl%29%29" :sig "-"}
"q@over@-" {:lang "q" :algo "over" :lib "-" :id 1 :doc "https://code.kx.com/q/ref/over/" :sig "-"}
"C++@reduce@<numeric>" {:lang "C++" :algo "reduce" :lib "<numeric>" :id 1 :doc "https://en.cppreference.com/w/cpp/algorithm/reduce" :sig "-"}
"Clojure@reduce@core" {:lang "Clojure" :algo "reduce" :lib "core" :id 1 :doc "https://clojuredocs.org/clojure.core/reduce" :sig "-"}
"CUDA@reduce@Thrust" {:lang "CUDA" :algo "reduce" :lib "Thrust" :id 1 :doc "https://thrust.github.io/doc/group__reductions_ga43eea9a000f912716189687306884fc7.html" :sig "-"}
"D@reduce@algorithm.iteration" {:lang "D" :algo "reduce" :lib "algorithm.iteration" :id 1 :doc "https://dlang.org/library/std/algorithm/iteration/reduce.html" :sig "-"}
"Elixir@reduce@Enum" {:lang "Elixir" :algo "reduce" :lib "Enum" :id 1 :doc "https://hexdocs.pm/elixir/Enum.html#reduce/2" :sig "-"}
"Kotlin@reduce@collections" {:lang "Kotlin" :algo "reduce" :lib "collections" :id 1 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/reduce.html" :sig "-"}
"Python@reduce@functools" {:lang "Python" :algo "reduce" :lib "functools" :id 1 :doc "https://docs.python.org/3/library/functools.html#functools.reduce" :sig "-"}
"Ruby@reduce@Enumerable" {:lang "Ruby" :algo "reduce" :lib "Enumerable" :id 1 :doc "https://apidock.com/ruby/Enumerable/reduce" :sig "-"}
"APL@¨ (each)@-" {:lang "APL" :algo "¨ (each)" :lib "-" :id 2 :doc "http://microapl.com/apl_help/ch_020_020_900.htm" :sig "-"}
"J@each@-" {:lang "J" :algo "each" :lib "-" :id 2 :doc "https://www.jsoftware.com/help/primer/each.htm" :sig "-"}
"q@each@-" {:lang "q" :algo "each" :lib "-" :id 2 :doc "https://code.kx.com/q/ref/each/" :sig "-"}
"Clojure@map@core" {:lang "Clojure" :algo "map" :lib "core" :id 2 :doc "https://clojuredocs.org/clojure.core/map" :sig "-"}
"D@map@algorithm.iteration" {:lang "D" :algo "map" :lib "algorithm.iteration" :id 2 :doc "https://dlang.org/library/std/algorithm/iteration/map.html" :sig "-"}
"Elixir@map@Enum" {:lang "Elixir" :algo "map" :lib "Enum" :id 2 :doc "https://hexdocs.pm/elixir/Enum.html#map/2" :sig "-"}
"Haskell@map@Data.List" {:lang "Haskell" :algo "map" :lib "Data.List" :id 2 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:map" :sig "-"}
"Kotlin@map@collections" {:lang "Kotlin" :algo "map" :lib "collections" :id 2 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/map.html" :sig "-"}
"Python@map@-" {:lang "Python" :algo "map" :lib "-" :id 2 :doc "https://docs.python.org/3/library/functions.html#map" :sig "-"}
"Racket@map@base" {:lang "Racket" :algo "map" :lib "base" :id 2 :doc "https://docs.racket-lang.org/reference/pairs.html?q=map#%28def._%28%28lib._racket%2Fprivate%2Fmap..rkt%29._map%29%29" :sig "-"}
"Ruby@map@Enumerable" {:lang "Ruby" :algo "map" :lib "Enumerable" :id 2 :doc "https://apidock.com/ruby/Enumerable/map" :sig "-"}
"Rust@map@trait.Iterator" {:lang "Rust" :algo "map" :lib "trait.Iterator" :id 2 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.map" :sig "-"}
"Scala@map@various" {:lang "Scala" :algo "map" :lib "various" :id 2 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#map[B](f:A=%3EB):scala.collection.immutable.Vector[B]" :sig "-"}
"C++@transform@<algorithm>" {:lang "C++" :algo "transform" :lib "<algorithm>" :id 2 :doc "https://en.cppreference.com/w/cpp/algorithm/transform" :sig "-"}
"APL@/ (compress)@-" {:lang "APL" :algo "/ (compress)" :lib "-" :id 3 :doc "http://microapl.com/apl_help/ch_020_020_840.htm" :sig "-"}
"Haskell@filter@Data.List" {:lang "Haskell" :algo "filter" :lib "Data.List" :id 3 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:filter" :sig "-"}
"J@\\ (prefix)@-" {:lang "J" :algo "\\ (prefix)" :lib "-" :id 4 :doc "https://code.jsoftware.com/wiki/Vocabulary/bslash" :sig "-"}
"APL@\\ (scan)@-" {:lang "APL" :algo "\\ (scan)" :lib "-" :id 4 :doc "http://microapl.com/apl_help/ch_020_020_820.htm" :sig "-"}
"Python@accumulate@itertools" {:lang "Python" :algo "accumulate" :lib "itertools" :id 4 :doc "https://docs.python.org/3/library/itertools.html#itertools.accumulate" :sig "-"}
"D@cumulativeFold@algorithm.iteration" {:lang "D" :algo "cumulativeFold" :lib "algorithm.iteration" :id 4 :doc "https://dlang.org/library/std/algorithm/iteration/cumulative_fold.html" :sig "-"}
"C++@inclusive_scan@<numeric>" {:lang "C++" :algo "inclusive_scan" :lib "<numeric>" :id 4 :doc "https://en.cppreference.com/w/cpp/algorithm/inclusive_scan" :sig "-"}
"CUDA@inclusive_scan@Thrust" {:lang "CUDA" :algo "inclusive_scan" :lib "Thrust" :id 4 :doc "https://thrust.github.io/doc/group__prefixsums_gafb24ad76101263038b0acaddc094d70a.html" :sig "-"}
"C++@partial_sum@<numeric>" {:lang "C++" :algo "partial_sum" :lib "<numeric>" :id 4 :doc "https://en.cppreference.com/w/cpp/algorithm/partial_sum" :sig "-"}
"Clojure@reductions@core" {:lang "Clojure" :algo "reductions" :lib "core" :id 4 :doc "https://clojuredocs.org/clojure.core/reductions" :sig "-"}
"q@scan@-" {:lang "q" :algo "scan" :lib "-" :id 4 :doc "https://code.kx.com/q/ref/over/" :sig "-"}
"Rust@scan@trait.Iterator" {:lang "Rust" :algo "scan" :lib "trait.Iterator" :id 4 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.scan" :sig "-"}
"Scala@scan@various" {:lang "Scala" :algo "scan" :lib "various" :id 4 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#scan[B%3E:A,That](z:B)(op:(B,B)=%3EB)(implicitcbf:scala.collection.generic.CanBuildFrom[Repr,B,That]):That" :sig "-"}
"Kotlin@runningReduce@collections" {:lang "Kotlin" :algo "runningReduce" :lib "collections" :id 4 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/running-reduce.html" :sig "-"}
"F#@scan@Seq" {:lang "F#" :algo "scan" :lib "Seq" :id 4 :doc "https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/sequences" :sig "-"}
"Elixir@scan@Enum" {:lang "Elixir" :algo "scan" :lib "Enum" :id 4 :doc "https://hexdocs.pm/elixir/Enum.html#scan/2" :sig "-"}
"Haskell@scanl1@Data.List" {:lang "Haskell" :algo "scanl1" :lib "Data.List" :id 4 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:scanl1" :sig "-"}
"Python@cumsum@RAPIDS cuDF" {:lang "Python" :algo "cumsum" :lib "RAPIDS cuDF" :id 4 :doc "https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.core.series.Series.cumsum" :sig "-"}
"Python@cumsum@pandas" {:lang "Python" :algo "cumsum" :lib "pandas" :id 4 :doc "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cumsum.html?highlight=cumsum#pandas.Series.cumsum" :sig "-"}
"APL@/ (n-wise reduce)@-" {:lang "APL" :algo "/ (n-wise reduce)" :lib "-" :id 5 :doc "http://microapl.com/apl_help/ch_020_020_800.htm" :sig "-"}
"C++@adjacent_difference@<numeric>" {:lang "C++" :algo "adjacent_difference" :lib "<numeric>" :id 5 :doc "https://en.cppreference.com/w/cpp/algorithm/adjacent_difference" :sig "-"}
"CUDA@adjacent_difference@Thrust" {:lang "CUDA" :algo "adjacent_difference" :lib "Thrust" :id 5 :doc "https://thrust.github.io/doc/group__transformations_gaa41d309b53fa03bf13fe35a184148400.html" :sig "-"}
"Haskell@mapAdjacent@Data.List.HT" {:lang "Haskell" :algo "mapAdjacent" :lib "Data.List.HT" :id 5 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:mapAdjacent" :sig "-"}
"q@prior@-" {:lang "q" :algo "prior" :lib "-" :id 5 :doc "https://code.kx.com/q/ref/prior/" :sig "-"}
"Kotlin@zipWithNext@collections" {:lang "Kotlin" :algo "zipWithNext" :lib "collections" :id 5 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/zip-with-next.html" :sig "-"}
"Python@pairwise@itertools" {:lang "Python" :algo "pairwise" :lib "itertools" :id 5 :doc "https://docs.python.org/3/library/itertools.html#itertools.pairwise" :sig "-"}
"APL@∘. (outer product)@-" {:lang "APL" :algo "∘. (outer product)" :lib "-" :id 6 :doc "http://microapl.com/apl_help/ch_020_020_890.htm" :sig "-"}
"Haskell@outerProduct@Data.List.HT" {:lang "Haskell" :algo "outerProduct" :lib "Data.List.HT" :id 6 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:outerProduct" :sig "-"}
"Clojure@outer-product@core.matrix" {:lang "Clojure" :algo "outer-product" :lib "core.matrix" :id 6 :doc "https://mikera.github.io/core.matrix/doc/clojure.core.matrix.html#var-outer-product" :sig "-"}
"APL@. (inner product)@-" {:lang "APL" :algo ". (inner product)" :lib "-" :id 7 :doc "http://microapl.com/apl_help/ch_020_020_880.htm" :sig "-"}
"C++@inner_product@<numeric>" {:lang "C++" :algo "inner_product" :lib "<numeric>" :id 7 :doc "https://en.cppreference.com/w/cpp/algorithm/inner_product" :sig "-"}
"Rust@.. / Range@ops" {:lang "Rust" :algo ".. / Range" :lib "ops" :id 8 :doc "https://doc.rust-lang.org/std/ops/struct.Range.html" :sig "-"}
"APL@⍳ (index generator)@-" {:lang "APL" :algo "⍳ (index generator)" :lib "-" :id 8 :doc "http://microapl.com/apl_help/ch_020_020_150.htm" :sig "-"}
"J@i. (integers)@-" {:lang "J" :algo "i. (integers)" :lib "-" :id 8 :doc "https://code.jsoftware.com/wiki/Vocabulary/idot" :sig "-"}
"C++@iota@<numeric>" {:lang "C++" :algo "iota" :lib "<numeric>" :id 8 :doc "https://en.cppreference.com/w/cpp/algorithm/iota" :sig "-"}
"D@iota@range" {:lang "D" :algo "iota" :lib "range" :id 8 :doc "https://dlang.org/library/std/range/iota.html" :sig "-"}
"Go@iota@-" {:lang "Go" :algo "iota" :lib "-" :id 8 :doc "https://github.com/golang/go/wiki/Iota" :sig "-"}
"Clojure@range@core" {:lang "Clojure" :algo "range" :lib "core" :id 8 :doc "https://clojuredocs.org/clojure.core/range" :sig "-"}
"Haskell@range@Data.List.HT" {:lang "Haskell" :algo "range" :lib "Data.List.HT" :id 8 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:range" :sig "-"}
"Python@range@-" {:lang "Python" :algo "range" :lib "-" :id 8 :doc "https://docs.python.org/3/library/functions.html#func-range" :sig "-"}
"Racket@range@base" {:lang "Racket" :algo "range" :lib "base" :id 8 :doc "https://docs.racket-lang.org/reference/pairs.html?q=enumerate#%28def._%28%28lib._racket%2Flist..rkt%29._range%29%29" :sig "-"}
"CUDA@sequence@Thrust" {:lang "CUDA" :algo "sequence" :lib "Thrust" :id 8 :doc "https://thrust.github.io/doc/group__transformations_ga233a3db0c5031023c8e9385acd4b9759.html" :sig "-"}
"q@til@-" {:lang "q" :algo "til" :lib "-" :id 8 :doc "https://code.kx.com/q/ref/til/" :sig "-"}
"Scala@until@Range" {:lang "Scala" :algo "until" :lib "Range" :id 8 :doc "https://www.scala-lang.org/api/2.12.3/scala/collection/immutable/Range.html" :sig "-"}
"D@enumerate@range" {:lang "D" :algo "enumerate" :lib "range" :id 9 :doc "https://dlang.org/library/std/range/enumerate.html" :sig "-"}
"Python@enumerate@-" {:lang "Python" :algo "enumerate" :lib "-" :id 9 :doc "https://docs.python.org/3/library/functions.html#enumerate" :sig "-"}
"Racket@enumerate@list-utils" {:lang "Racket" :algo "enumerate" :lib "list-utils" :id 9 :doc "https://docs.racket-lang.org/list-utils/index.html?q=freq#%28def._%28%28lib._list-utils%2Fmain..rkt%29._enumerate%29%29" :sig "-"}
"Rust@enumerate@trait.Iterator" {:lang "Rust" :algo "enumerate" :lib "trait.Iterator" :id 9 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.enumerate" :sig "-"}
"Haskell@indexed@Data.List.Index" {:lang "Haskell" :algo "indexed" :lib "Data.List.Index" :id 9 :doc "https://hackage.haskell.org/package/ilist-0.4.0.0/docs/Data-List-Index.html#v:indexed" :sig "-"}
"Racket@indexed@data-collections" {:lang "Racket" :algo "indexed" :lib "data-collections" :id 9 :doc "https://docs.racket-lang.org/collections/collections-api.html?q=chunk-while#%28def._%28%28lib._data%2Fcollection..rkt%29._indexed%29%29" :sig "-"}
"Clojure@map-indexed*@core" {:lang "Clojure" :algo "map-indexed*" :lib "core" :id 9 :doc "https://clojuredocs.org/clojure.core/map-indexed" :sig "-"}
"C#@Select@Enumerable" {:lang "C#" :algo "Select" :lib "Enumerable" :id 9 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.select" :sig "-"}
"Elixir@with_index@Enum" {:lang "Elixir" :algo "with_index" :lib "Enum" :id 9 :doc "https://hexdocs.pm/elixir/Enum.html#with_index/2" :sig "-"}
"Ruby@with_index@Enumerable" {:lang "Ruby" :algo "with_index" :lib "Enumerable" :id 9 :doc "https://apidock.com/ruby/Enumerator/with_index" :sig "-"}
"Kotlin@withIndex@collections" {:lang "Kotlin" :algo "withIndex" :lib "collections" :id 9 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/with-index.html" :sig "-"}
"Scala@zipWithIndex@various" {:lang "Scala" :algo "zipWithIndex" :lib "various" :id 9 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#zipWithIndex:scala.collection.immutable.Vector[(A,Int)]" :sig "-"}
"Haskell@partition@Data.List" {:lang "Haskell" :algo "partition" :lib "Data.List" :id 10 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Data-List.html#v:partition" :sig "-"}
"Kotlin@partition@collections" {:lang "Kotlin" :algo "partition" :lib "collections" :id 10 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/partition.html" :sig "-"}
"Python@partition@more-itertools" {:lang "Python" :algo "partition" :lib "more-itertools" :id 10 :doc "https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partition" :sig "-"}
"Racket@partition@base" {:lang "Racket" :algo "partition" :lib "base" :id 10 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._partition%29%29" :sig "-"}
"Ruby@partition@Enumerable" {:lang "Ruby" :algo "partition" :lib "Enumerable" :id 10 :doc "https://apidock.com/ruby/Enumerable/partition" :sig "-"}
"Rust@partition@trait.Iterator" {:lang "Rust" :algo "partition" :lib "trait.Iterator" :id 10 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.partition" :sig "-"}
"C++@partition_copy@<algorithm>" {:lang "C++" :algo "partition_copy" :lib "<algorithm>" :id 10 :doc "https://en.cppreference.com/w/cpp/algorithm/partition_copy" :sig "-"}
"Elixir@split_with@Enum" {:lang "Elixir" :algo "split_with" :lib "Enum" :id 10 :doc "https://hexdocs.pm/elixir/Enum.html#split_with/2" :sig "-"}
"Clojure@partition@core" {:lang "Clojure" :algo "partition" :lib "core" :id 11 :doc "https://clojuredocs.org/clojure.core/partition" :sig "-"}
"C++@partition@<algorithm>" {:lang "C++" :algo "partition" :lib "<algorithm>" :id 13 :doc "https://en.cppreference.com/w/cpp/algorithm/partition" :sig "-"}
"CUDA@partition@Thrust" {:lang "CUDA" :algo "partition" :lib "Thrust" :id 13 :doc "https://thrust.github.io/doc/group__partitioning_gac5cdbb402c5473ca92e95bc73ecaf13c.html" :sig "-"}
"D@partition@algorithm.sorting" {:lang "D" :algo "partition" :lib "algorithm.sorting" :id 13 :doc "https://dlang.org/library/std/algorithm/sorting/partition.html" :sig "-"}
"Shell@uniq@-" {:lang "Shell" :algo "uniq" :lib "-" :id 14 :doc "https://ss64.com/bash/uniq.html" :sig "-"}
"D@uniq@algorithm.iteration" {:lang "D" :algo "uniq" :lib "algorithm.iteration" :id 14 :doc "https://dlang.org/phobos/std_algorithm_iteration.html#uniq" :sig "-"}
"C++@unique@<algorithm>" {:lang "C++" :algo "unique" :lib "<algorithm>" :id 14 :doc "https://en.cppreference.com/w/cpp/algorithm/unique" :sig "-"}
"C#@Distinct@Enumerable" {:lang "C#" :algo "Distinct" :lib "Enumerable" :id 15 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.distinct" :sig "-"}
"Clojure@distinct@core" {:lang "Clojure" :algo "distinct" :lib "core" :id 15 :doc "https://clojuredocs.org/clojure.core/distinct" :sig "-"}
"F#@Distinct@Seq" {:lang "F#" :algo "Distinct" :lib "Seq" :id 15 :doc "https://www.dotnetperls.com/duplicates-fs" :sig "-"}
"Kotlin@distinct@collections" {:lang "Kotlin" :algo "distinct" :lib "collections" :id 15 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/distinct.html" :sig "-"}
"Scala@distinct@various" {:lang "Scala" :algo "distinct" :lib "various" :id 15 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#distinct:Repr" :sig "-"}
"Haskell@nub@Data.List" {:lang "Haskell" :algo "nub" :lib "Data.List" :id 15 :doc "https://hackage.haskell.org/package/base-4.18.1.0/docs/Data-List.html#v:nub" :sig "-"}
"Haskell@sortUniq@Data.List.Unique" {:lang "Haskell" :algo "sortUniq" :lib "Data.List.Unique" :id 15 :doc "https://hackage.haskell.org/package/Unique-0.4.7.7/docs/Data-List-Unique.html#v:sortUniq" :sig "-"}
"Elixir@uniq@Enum" {:lang "Elixir" :algo "uniq" :lib "Enum" :id 15 :doc "https://hexdocs.pm/elixir/Enum.html#uniq/1" :sig "-"}
"Ruby@uniq@Array" {:lang "Ruby" :algo "uniq" :lib "Array" :id 15 :doc "https://apidock.com/ruby/Array/uniq" :sig "-"}
"Rust@unique@itertools" {:lang "Rust" :algo "unique" :lib "itertools" :id 15 :doc "https://docs.rs/itertools/0.7.6/itertools/structs/struct.Unique.html" :sig "-"}
"J@\": (default format)@-" {:lang "J" :algo "\": (default format)" :lib "-" :id 16 :doc "https://code.jsoftware.com/wiki/Vocabulary/quoteco" :sig "-"}
"APL@⍕ (format)@-" {:lang "APL" :algo "⍕ (format)" :lib "-" :id 16 :doc "http://microapl.com/apl_help/ch_020_020_680.htm" :sig "-"}
"Haskell@show@Prelude" {:lang "Haskell" :algo "show" :lib "Prelude" :id 16 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:show" :sig "-"}
"Clojure@str@core" {:lang "Clojure" :algo "str" :lib "core" :id 16 :doc "https://clojuredocs.org/clojure.core/str" :sig "-"}
"Python@str@-" {:lang "Python" :algo "str" :lib "-" :id 16 :doc "https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str" :sig "-"}
"q@string@-" {:lang "q" :algo "string" :lib "-" :id 16 :doc "https://code.kx.com/v2/ref/string/" :sig "-"}
"C++@to_string@<string>" {:lang "C++" :algo "to_string" :lib "<string>" :id 16 :doc "https://en.cppreference.com/w/cpp/string/basic_string/to_string" :sig "-"}
"Elixir@to_string@various" {:lang "Elixir" :algo "to_string" :lib "various" :id 16 :doc "https://hexdocs.pm/elixir/Integer.html#to_string/1" :sig "-"}
"Rust@to_string@string" {:lang "Rust" :algo "to_string" :lib "string" :id 16 :doc "https://doc.rust-lang.org/std/string/trait.ToString.html" :sig "-"}
"Scala@toString@various" {:lang "Scala" :algo "toString" :lib "various" :id 16 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#toString():String" :sig "-"}
"D@canFind@algorithm.searching" {:lang "D" :algo "canFind" :lib "algorithm.searching" :id 17 :doc "https://dlang.org/library/std/algorithm/searching/can_find.can_find.html" :sig "-"}
"Kotlin@contains@collections" {:lang "Kotlin" :algo "contains" :lib "collections" :id 17 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/contains.html" :sig "-"}
"Scala@contains@various" {:lang "Scala" :algo "contains" :lib "various" :id 17 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#contains[A1%3E:A](elem:A1):Boolean" :sig "-"}
"Clojure@contains?@core" {:lang "Clojure" :algo "contains?" :lib "core" :id 17 :doc "https://clojuredocs.org/clojure.core/contains_q" :sig "-"}
"Haskell@elem@Prelude" {:lang "Haskell" :algo "elem" :lib "Prelude" :id 17 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:elem" :sig "-"}
"Python@in@-" {:lang "Python" :algo "in" :lib "-" :id 17 :doc "https://www.programiz.com/python-programming/keyword-list#in" :sig "-"}
"Ruby@include?@various" {:lang "Ruby" :algo "include?" :lib "various" :id 17 :doc "https://apidock.com/ruby/Array/include%3F" :sig "-"}
"Racket@member*@base" {:lang "Racket" :algo "member*" :lib "base" :id 17 :doc "https://docs.racket-lang.org/reference/pairs.html?q=map#%28def._%28%28lib._racket%2Fprivate%2Fbase..rkt%29._member%29%29" :sig "-"}
"Elixir@member?@Enum" {:lang "Elixir" :algo "member?" :lib "Enum" :id 17 :doc "https://hexdocs.pm/elixir/Enum.html#member?/2" :sig "-"}
"APL@⍸ (where)@-" {:lang "APL" :algo "⍸ (where)" :lib "-" :id 18 :doc "https://help.dyalog.com/17.1/#Language/Primitive%20Functions/Where.htm#kanchor3608" :sig "-"}
"q@where@-" {:lang "q" :algo "where" :lib "-" :id 18 :doc "https://code.kx.com/q4m3/A_Built-in_Functions/#a107-where" :sig "-"}
"Haskell@count@Data.List.Unique" {:lang "Haskell" :algo "count" :lib "Data.List.Unique" :id 19 :doc "https://hackage.haskell.org/package/Unique-0.4.7.7/docs/Data-List-Unique.html#v:count" :sig "-"}
"Python@Counter*@collections" {:lang "Python" :algo "Counter*" :lib "collections" :id 19 :doc "https://docs.python.org/3/library/collections.html#collections.Counter" :sig "-"}
"Clojure@frequencies@core" {:lang "Clojure" :algo "frequencies" :lib "core" :id 19 :doc "https://clojuredocs.org/clojure.core/frequencies" :sig "-"}
"Racket@frequencies@list-utils" {:lang "Racket" :algo "frequencies" :lib "list-utils" :id 19 :doc "https://docs.racket-lang.org/list-utils/index.html?q=freq#%28def._%28%28lib._list-utils%2Fmain..rkt%29._frequencies%29%29" :sig "-"}
"Python@value_counts@cuDF" {:lang "Python" :algo "value_counts" :lib "cuDF" :id 19 :doc "https://docs.rapids.ai/api/cudf/stable/api.html#cudf.core.series.Series.value_counts" :sig "-"}
"Python@value_counts@pandas" {:lang "Python" :algo "value_counts" :lib "pandas" :id 19 :doc "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html" :sig "-"}
"D@any@algorithm.searching" {:lang "D" :algo "any" :lib "algorithm.searching" :id 20 :doc "https://dlang.org/library/std/algorithm/searching/any.html" :sig "-"}
"Haskell@any@Prelude" {:lang "Haskell" :algo "any" :lib "Prelude" :id 20 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:any" :sig "-"}
"Kotlin@any@collections" {:lang "Kotlin" :algo "any" :lib "collections" :id 20 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/any.html" :sig "-"}
"Python@any@-" {:lang "Python" :algo "any" :lib "-" :id 20 :doc "https://docs.python.org/3/library/functions.html#any" :sig "-"}
"Rust@any@trait.Iterator" {:lang "Rust" :algo "any" :lib "trait.Iterator" :id 20 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.any" :sig "-"}
"Elixir@any?@Enum" {:lang "Elixir" :algo "any?" :lib "Enum" :id 20 :doc "https://hexdocs.pm/elixir/Enum.html#any?/2" :sig "-"}
"Ruby@any?@Enumerable" {:lang "Ruby" :algo "any?" :lib "Enumerable" :id 20 :doc "https://apidock.com/ruby/Enumerable/any%3F" :sig "-"}
"C++@any_of@<algorithm>" {:lang "C++" :algo "any_of" :lib "<algorithm>" :id 20 :doc "https://en.cppreference.com/w/cpp/algorithm/all_any_none_of" :sig "-"}
"F#@exists@List" {:lang "F#" :algo "exists" :lib "List" :id 20 :doc "https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/lists" :sig "-"}
"Scala@exists@various" {:lang "Scala" :algo "exists" :lib "various" :id 20 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html" :sig "-"}
"Racket@memf*@base" {:lang "Racket" :algo "memf*" :lib "base" :id 20 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Fprivate%2Flist..rkt%29._memf%29%29" :sig "-"}
"Clojure@some@core" {:lang "Clojure" :algo "some" :lib "core" :id 20 :doc "https://clojuredocs.org/clojure.core/some" :sig "-"}
"APL@, (catenate)@-" {:lang "APL" :algo ", (catenate)" :lib "-" :id 21 :doc "http://microapl.com/apl_help/ch_020_020_490.htm" :sig "-"}
"Haskell@concat@Prelude" {:lang "Haskell" :algo "concat" :lib "Prelude" :id 21 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Prelude.html#v:concat" :sig "-"}
"Racket@join@base" {:lang "Racket" :algo "join" :lib "base" :id 21 :doc "https://docs.racket-lang.org/algebraic/class_base.html?q=some#%28def._%28%28lib._algebraic%2Fcontrol%2Fmonad..rkt%29._join%29%29" :sig "-"}
"Ruby@join@various" {:lang "Ruby" :algo "join" :lib "various" :id 21 :doc "https://apidock.com/ruby/Array/join" :sig "-"}
"Clojure@join@string" {:lang "Clojure" :algo "join" :lib "string" :id 21 :doc "https://clojuredocs.org/clojure.string/join" :sig "-"}
"D@join@array" {:lang "D" :algo "join" :lib "array" :id 21 :doc "https://dlang.org/library/std/array/join.html" :sig "-"}
"Rust@join@Vec" {:lang "Rust" :algo "join" :lib "Vec" :id 21 :doc "https://doc.rust-lang.org/std/vec/struct.Vec.html#method.join" :sig "-"}
"Go@join@strings" {:lang "Go" :algo "join" :lib "strings" :id 21 :doc "https://golang.org/pkg/strings/#Join" :sig "-"}
"C#@join@String" {:lang "C#" :algo "join" :lib "String" :id 21 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.string.join?view=netframework-4.8" :sig "-"}
"F#@join@String" {:lang "F#" :algo "join" :lib "String" :id 21 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.string.join?view=netframework-4.8" :sig "-"}
"Python@join*@str" {:lang "Python" :algo "join*" :lib "str" :id 21 :doc "https://docs.python.org/3/library/stdtypes.html#str.join" :sig "-"}
"Kotlin@joinTo@collections" {:lang "Kotlin" :algo "joinTo" :lib "collections" :id 21 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/join-to.html" :sig "-"}
"Python@zip(*)@-" {:lang "Python" :algo "zip(*)" :lib "-" :id 22 :doc "https://docs.python.org/3/library/functions.html#zip" :sig "-"}
"Python@transpose@pandas" {:lang "Python" :algo "transpose" :lib "pandas" :id 22 :doc "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html" :sig "-"}
"Ruby@transpose@Array" {:lang "Ruby" :algo "transpose" :lib "Array" :id 22 :doc "https://apidock.com/ruby/Array/transpose" :sig "-"}
"APL@⍉ (transpose)@-" {:lang "APL" :algo "⍉ (transpose)" :lib "-" :id 22 :doc "http://microapl.com/apl_help/ch_020_020_540.htm" :sig "-"}
"Haskell@transpose@Data.List" {:lang "Haskell" :algo "transpose" :lib "Data.List" :id 22 :doc "https://hackage.haskell.org/package/base-4.12.0.0/docs/Data-List.html#v:transpose" :sig "-"}
"q@flip@- " {:lang "q" :algo "flip" :lib "- " :id 22 :doc "https://code.kx.com/v2/ref/flip/" :sig "-"}
"Clojure@transpose@core.matrix" {:lang "Clojure" :algo "transpose" :lib "core.matrix" :id 22 :doc "https://mikera.github.io/core.matrix/doc/clojure.core.matrix.html#var-transpose" :sig "-"}
"Elixir@transpose@Matrix" {:lang "Elixir" :algo "transpose" :lib "Matrix" :id 22 :doc "https://hexdocs.pm/matrix/Matrix.html#transpose/1" :sig "-"}
"Scala@transpose@various" {:lang "Scala" :algo "transpose" :lib "various" :id 22 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#transpose" :sig "-"}
"D@transposed@range" {:lang "D" :algo "transposed" :lib "range" :id 22 :doc "https://dlang.org/library/std/range/transposed.html" :sig "-"}
"Python@transpose@RAPIDS cuDF" {:lang "Python" :algo "transpose" :lib "RAPIDS cuDF" :id 22 :doc "https://docs.rapids.ai/api/cudf/stable/api.html#cudf.core.dataframe.DataFrame.transpose" :sig "-"}
"APL@⊆ (partition)@-" {:lang "APL" :algo "⊆ (partition)" :lib "-" :id 23 :doc "http://microapl.com/apl_help/ch_020_020_590.htm" :sig "-"}
"Haskell@groupBy@Data.List.HT" {:lang "Haskell" :algo "groupBy" :lib "Data.List.HT" :id 23 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:groupBy" :sig "-"}
"Python@groupby@itertools" {:lang "Python" :algo "groupby" :lib "itertools" :id 23 :doc "https://docs.python.org/3/library/itertools.html#itertools.groupby" :sig "-"}
"Clojure@partition-by@core" {:lang "Clojure" :algo "partition-by" :lib "core" :id 23 :doc "https://clojuredocs.org/clojure.core/partition-by" :sig "-"}
"Ruby@chunk_while@Enumerable" {:lang "Ruby" :algo "chunk_while" :lib "Enumerable" :id 23 :doc "https://apidock.com/ruby/Enumerable/chunk_while" :sig "-"}
"D@chunkBy@algorith.iteration" {:lang "D" :algo "chunkBy" :lib "algorith.iteration" :id 23 :doc "https://dlang.org/library/std/algorithm/iteration/chunk_by.html" :sig "-"}
"Elixir@chunk_while@Enum" {:lang "Elixir" :algo "chunk_while" :lib "Enum" :id 23 :doc "https://hexdocs.pm/elixir/Enum.html#chunk_while/4" :sig "-"}
"Ruby@slice_when@Enumerable" {:lang "Ruby" :algo "slice_when" :lib "Enumerable" :id 23 :doc "https://apidock.com/ruby/v2_5_5/Enumerable/slice_when" :sig "-"}
"C++@chunk_by@range-v3" {:lang "C++" :algo "chunk_by" :lib "range-v3" :id 23 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-"}
"C++@chunk_by@<ranges>" {:lang "C++" :algo "chunk_by" :lib "<ranges>" :id 23 :doc "https://en.cppreference.com/w/cpp/ranges/chunk_by_view" :sig "-"}
"Haskell@repeat@Prelude" {:lang "Haskell" :algo "repeat" :lib "Prelude" :id 24 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:repeat" :sig "-"}
"Clojure@repeat@core" {:lang "Clojure" :algo "repeat" :lib "core" :id 24 :doc "https://clojuredocs.org/clojure.core/repeat" :sig "-"}
"C++@repeat@range-v3" {:lang "C++" :algo "repeat" :lib "range-v3" :id 24 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-"}
"D@repeat@range" {:lang "D" :algo "repeat" :lib "range" :id 24 :doc "https://dlang.org/library/std/range/repeat.html" :sig "-"}
"Rust@repeat@Iter" {:lang "Rust" :algo "repeat" :lib "Iter" :id 24 :doc "https://doc.rust-lang.org/std/iter/struct.Repeat.html" :sig "-"}
"Haskell@replicate@Prelude" {:lang "Haskell" :algo "replicate" :lib "Prelude" :id 25 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:replicate" :sig "-"}
"C++@repeat_n@range-v3" {:lang "C++" :algo "repeat_n" :lib "range-v3" :id 25 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-"}
"C#@repeat@Enumerable" {:lang "C#" :algo "repeat" :lib "Enumerable" :id 25 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.repeat?view=netcore-3.1" :sig "-"}
"APL@\\ (expand)@-" {:lang "APL" :algo "\\ (expand)" :lib "-" :id 25 :doc "http://microapl.com/apl_help/ch_020_020_860.htm" :sig "-"}
"C++@transform*@<algorithm>" {:lang "C++" :algo "transform*" :lib "<algorithm>" :id 30 :doc "https://en.cppreference.com/w/cpp/algorithm/transform" :sig "-"}
"Haskell@zipWith@Prelude" {:lang "Haskell" :algo "zipWith" :lib "Prelude" :id 30 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:zipWith" :sig "-"}
"Clojure@map*@core" {:lang "Clojure" :algo "map*" :lib "core" :id 30 :doc "https://clojuredocs.org/clojure.core/map" :sig "-"}
"Racket@map*@base" {:lang "Racket" :algo "map*" :lib "base" :id 30 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Fprivate%2Fmap..rkt%29._map%29%29" :sig "-"}
"OCaml@map2@List" {:lang "OCaml" :algo "map2" :lib "List" :id 30 :doc "https://caml.inria.fr/pub/docs/manual-ocaml/libref/List.html" :sig "-"}
"F#@map2@Seq" {:lang "F#" :algo "map2" :lib "Seq" :id 30 :doc "https://msdn.microsoft.com/visualfsharpdocs/conceptual/list.map2%5b%27t1%2c%27t2%2c%27u%5d-function-%5bfsharp%5d" :sig "-"}
"C++@filter@range-v3" {:lang "C++" :algo "filter" :lib "range-v3" :id 3 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-"}
"C#@Where@Enumerable" {:lang "C#" :algo "Where" :lib "Enumerable" :id 3 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.where?view=netcore-3.1" :sig "-"}
"J@copy@-" {:lang "J" :algo "copy" :lib "-" :id 3 :doc "https://code.jsoftware.com/wiki/Vocabulary/number#dyadic" :sig "-"}
"Java@filter@Streams" {:lang "Java" :algo "filter" :lib "Streams" :id 3 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#filter-java.util.function.Predicate-" :sig "-"}
"OCaml@filter@List" {:lang "OCaml" :algo "filter" :lib "List" :id 3 :doc "https://ocaml.org/releases/4.10/htmlman/libref/List.html" :sig "-"}
"Clojure@filter@core" {:lang "Clojure" :algo "filter" :lib "core" :id 3 :doc "https://clojuredocs.org/clojure.core/filter" :sig "-"}
"Racket@filter@base" {:lang "Racket" :algo "filter" :lib "base" :id 3 :doc "https://docs.racket-lang.org/reference/pairs.html?q=filter#%28def._%28%28lib._racket%2Fprivate%2Flist..rkt%29._filter%29%29" :sig "-"}
"F#@filter@List" {:lang "F#" :algo "filter" :lib "List" :id 3 :doc "https://msdn.microsoft.com/visualfsharpdocs/conceptual/list.filter%5b%27t%5d-function-%5bfsharp%5d" :sig "-"}
"D@filter@algorithm" {:lang "D" :algo "filter" :lib "algorithm" :id 3 :doc "https://dlang.org/library/std/algorithm/iteration/filter.html" :sig "-"}
"Ruby@filter@Array" {:lang "Ruby" :algo "filter" :lib "Array" :id 3 :doc "https://apidock.com/ruby/Array/filter" :sig "-"}
"Python@filter@-" {:lang "Python" :algo "filter" :lib "-" :id 3 :doc "https://docs.python.org/3/library/functions.html#filter" :sig "-"}
"Kotlin@filter@collections" {:lang "Kotlin" :algo "filter" :lib "collections" :id 3 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/filter.html" :sig "-"}
"Scala@filter@various" {:lang "Scala" :algo "filter" :lib "various" :id 3 :doc "https://www.scala-lang.org/api/2.12.4/scala/collection/immutable/Vector.html#filter(p:A=%3EBoolean):Repr" :sig "-"}
"Rust@filter@iter" {:lang "Rust" :algo "filter" :lib "iter" :id 3 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.filter" :sig "-"}
"Elixir@filter@Enum" {:lang "Elixir" :algo "filter" :lib "Enum" :id 3 :doc "https://hexdocs.pm/elixir/Enum.html#filter/2" :sig "-"}
"D@cartesianProduct@std.algorithm.setops" {:lang "D" :algo "cartesianProduct" :lib "std.algorithm.setops" :id 100 :doc "https://dlang.org/library/std/algorithm/setops/cartesian_product.html" :sig "-"}
"D@dotProduct@std.numeric" {:lang "D" :algo "dotProduct" :lib "std.numeric" :id 7 :doc "https://dlang.org/library/std/numeric/dot_product.html" :sig "-"}
"C++@cartesian_product@range-v3" {:lang "C++" :algo "cartesian_product" :lib "range-v3" :id 100 :doc "https://ericniebler.github.io/range-v3/index.html#tutorial-views" :sig "-"}
"Python@product@itertools" {:lang "Python" :algo "product" :lib "itertools" :id 100 :doc "https://docs.python.org/3/library/itertools.html#itertools.product" :sig "-"}
"q@cross@-" {:lang "q" :algo "cross" :lib "-" :id 100 :doc "https://code.kx.com/q4m3/A_Built-in_Functions/#a18-cross" :sig "-"}
"J@. (Matrix Product)@-" {:lang "J" :algo ". (Matrix Product)" :lib "-" :id 7 :doc "https://code.jsoftware.com/wiki/Vocabulary/dot#dyadic" :sig "-"}
"R@outer@-" {:lang "R" :algo "outer" :lib "-" :id 6 :doc "https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/outer" :sig "-"}
"Ruby@product@Array" {:lang "Ruby" :algo "product" :lib "Array" :id 100 :doc "https://apidock.com/ruby/Array/product" :sig "-"}
"Racket@cartesian-product@base" {:lang "Racket" :algo "cartesian-product" :lib "base" :id 100 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._cartesian-product%29%29" :sig "-"}
"Rust@cartesian_product@trait.itertools" {:lang "Rust" :algo "cartesian_product" :lib "trait.itertools" :id 100 :doc "https://docs.rs/itertools/0.8.2/itertools/trait.Itertools.html#method.cartesian_product" :sig "-"}
"C++@all_of@<algorithm>" {:lang "C++" :algo "all_of" :lib "<algorithm>" :id 31 :doc "https://en.cppreference.com/w/cpp/algorithm/all_any_none_of" :sig "-"}
"Haskell@all@Prelude" {:lang "Haskell" :algo "all" :lib "Prelude" :id 31 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/GHC-List.html#v:all" :sig "-"}
"Clojure@every?@core" {:lang "Clojure" :algo "every?" :lib "core" :id 31 :doc "https://clojuredocs.org/clojure.core/every_q" :sig "-"}
"Kotlin@all@sequences" {:lang "Kotlin" :algo "all" :lib "sequences" :id 31 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/all.html" :sig "-"}
"Rust@all@iter" {:lang "Rust" :algo "all" :lib "iter" :id 31 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.all" :sig "-"}
"Scala@forall@various" {:lang "Scala" :algo "forall" :lib "various" :id 31 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.all" :sig "-"}
"Elixir@all?@Enum" {:lang "Elixir" :algo "all?" :lib "Enum" :id 31 :doc "https://hexdocs.pm/elixir/Enum.html#all?/2" :sig "-"}
"Python@all@-" {:lang "Python" :algo "all" :lib "-" :id 31 :doc "https://docs.python.org/3/library/functions.html#all" :sig "-"}
"Ruby@all@Enumerable" {:lang "Ruby" :algo "all" :lib "Enumerable" :id 31 :doc "https://apidock.com/ruby/Enumerable/all%3F" :sig "-"}
"Java@allMatch@Streams" {:lang "Java" :algo "allMatch" :lib "Streams" :id 31 :doc "https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#allMatch-java.util.function.Predicate-" :sig "-"}
"D@all@std.algorithm" {:lang "D" :algo "all" :lib "std.algorithm" :id 31 :doc "https://dlang.org/library/std/algorithm/searching/all.all.html" :sig "-"}
"Swift@allSatisfy@" {:lang "Swift" :algo "allSatisfy" :lib "" :id 31 :doc "https://developer.apple.com/documentation/swift/array/2994715-allsatisfy" :sig "-"}
"OCaml@for_all@List" {:lang "OCaml" :algo "for_all" :lib "List" :id 31 :doc "https://caml.inria.fr/pub/docs/manual-ocaml/libref/List.html" :sig "-"}
"C#@All@Linq" {:lang "C#" :algo "All" :lib "Linq" :id 31 :doc "https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.all?view=netcore-3.1" :sig "-"}
"JavaScript@every@" {:lang "JavaScript" :algo "every" :lib "" :id 31 :doc "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/every" :sig "-"}
"Elm@all@List" {:lang "Elm" :algo "all" :lib "List" :id 31 :doc "https://package.elm-lang.org/packages/elm/core/latest/List#all" :sig "-"}
"Julia@all@Base" {:lang "Julia" :algo "all" :lib "Base" :id 31 :doc "https://docs.julialang.org/en/v1/base/collections/#Base.all-Tuple{AbstractArray,Any}" :sig "-"}
"Haskell@++@Prelude" {:lang "Haskell" :algo "++" :lib "Prelude" :id 42 :doc "https://hackage.haskell.org/package/base-4.14.0.0/docs/Prelude.html#v:-43--43-" :sig "-"}
"Racket@append@base" {:lang "Racket" :algo "append" :lib "base" :id 42 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28quote._~23~25kernel%29._append%29%29" :sig "-"}
"Python@append@list" {:lang "Python" :algo "append" :lib "list" :id 42 :doc "https://docs.python.org/3/tutorial/introduction.html#lists" :sig "-"}
"Clojure@concat@List" {:lang "Clojure" :algo "concat" :lib "List" :id 42 :doc "https://clojuredocs.org/clojure.core/concat" :sig "-"}
"C++@join@Ranges" {:lang "C++" :algo "join" :lib "Ranges" :id 42 :doc "https://en.cppreference.com/w/cpp/ranges" :sig "-"}
"Swift@+@Array" {:lang "Swift" :algo "+" :lib "Array" :id 42 :doc "https://developer.apple.com/documentation/swift/array/2963457" :sig "-"}
"Rust@chain@iter" {:lang "Rust" :algo "chain" :lib "iter" :id 42 :doc "https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.chain" :sig "-"}
"Python@chain@itertools" {:lang "Python" :algo "chain" :lib "itertools" :id 42 :doc "https://docs.python.org/3/library/itertools.html#itertools.chain" :sig "-"}
"Julia@vcat@Base" {:lang "Julia" :algo "vcat" :lib "Base" :id 42 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.vcat" :sig "-"}
"Elixir@concat@Enum" {:lang "Elixir" :algo "concat" :lib "Enum" :id 42 :doc "https://hexdocs.pm/elixir/Enum.html#concat/2" :sig "-"}
"Elixir@++@Kernel" {:lang "Elixir" :algo "++" :lib "Kernel" :id 42 :doc "https://hexdocs.pm/elixir/Kernel.html#++/2" :sig "-"}
"F#@allPairs@List" {:lang "F#" :algo "allPairs" :lib "List" :id 100 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-listmodule.html#allPairs" :sig "-"}
"Julia@product@IterTools" {:lang "Julia" :algo "product" :lib "IterTools" :id 100 :doc "https://juliacollections.github.io/IterTools.jl/v0.2.1/#product(xs...)-1" :sig "-"}
"APL@,∘.,@" {:lang "APL" :algo ",∘.," :lib "" :id 100 :doc "" :sig "-"}
"SQL@CROSS JOIN@" {:lang "SQL" :algo "CROSS JOIN" :lib "" :id 100 :doc "https://www.w3resource.com/sql/joins/cross-join.php" :sig "-"}
"Python@all_equal@more-itertools" {:lang "Python" :algo "all_equal" :lib "more-itertools" :id 51 :doc "https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.all_equal" :sig "-"}
"Clojure@apply = @" {:lang "Clojure" :algo "apply = " :lib "" :id 51 :doc "" :sig "-"}
"Racket@apply = @" {:lang "Racket" :algo "apply = " :lib "" :id 51 :doc "" :sig "-"}
"Haskell@allEqual@Data.List.HT" {:lang "Haskell" :algo "allEqual" :lib "Data.List.HT" :id 51 :doc "https://hackage.haskell.org/package/utility-ht-0.0.15/docs/Data-List-HT.html#v:allEqual" :sig "-"}
"JavaScript@allEqual@bbo" {:lang "JavaScript" :algo "allEqual" :lib "bbo" :id 51 :doc "https://tnfe.github.io/bbo/#allequal" :sig "-"}
"C++@rotate@<algorithm>" {:lang "C++" :algo "rotate" :lib "<algorithm>" :id 4104 :doc "https://en.cppreference.com/w/cpp/algorithm/rotate" :sig "-"}
"APL@⌽ (rotate)@" {:lang "APL" :algo "⌽ (rotate)" :lib "" :id 4104 :doc "https://microapl.com/apl_help/ch_020_020_520.htm" :sig "-"}
"Ruby@rotate@Array" {:lang "Ruby" :algo "rotate" :lib "Array" :id 4104 :doc "https://apidock.com/ruby/v2_5_5/Array/rotate" :sig "-"}
"Rust@rotate_left@Slice" {:lang "Rust" :algo "rotate_left" :lib "Slice" :id 4104 :doc "https://doc.rust-lang.org/stable/std/primitive.slice.html#method.rotate_left" :sig "-"}
"Julia@circshift@" {:lang "Julia" :algo "circshift" :lib "" :id 4104 :doc "https://docs.julialang.org/en/v1/base/arrays/#Base.circshift" :sig "-"}
"q@rotate@Core" {:lang "q" :algo "rotate" :lib "Core" :id 4104 :doc "https://code.kx.com/q4m3/A_Built-in_Functions/#a79-rotate" :sig "-"}
"Rust@dedup@itertools" {:lang "Rust" :algo "dedup" :lib "itertools" :id 15 :doc "https://docs.rs/itertools/0.7.6/itertools/structs/struct.Dedup.html" :sig "-"}
"Racket@remove-duplicates@" {:lang "Racket" :algo "remove-duplicates" :lib "" :id 15 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket/list..rkt%29._remove-duplicates%29%29" :sig "-"}
"OCaml@sort_uniq@List" {:lang "OCaml" :algo "sort_uniq" :lib "List" :id 15 :doc "https://ocaml.org/api/List.html#VALsort_uniq" :sig "-"}
"Elm@dedup@List" {:lang "Elm" :algo "dedup" :lib "List" :id 14 :doc "https://package.elm-lang.org/packages/mgold/elm-nonempty-list/latest/List-Nonempty#dedup" :sig "-"}
"Elm@uniq@List" {:lang "Elm" :algo "uniq" :lib "List" :id 15 :doc "https://package.elm-lang.org/packages/mgold/elm-nonempty-list/latest/List-Nonempty#uniq" :sig "-"}
"F#@distinct@Seq" {:lang "F#" :algo "distinct" :lib "Seq" :id 15 :doc "https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-seqmodule.html#distinct" :sig "-"}
"Pharo@removeDuplicates@OrderedCollection" {:lang "Pharo" :algo "removeDuplicates" :lib "OrderedCollection" :id 15 :doc "" :sig "-"}
"BQN@ ` (scan) @-" {:lang "BQN" :algo " ` (scan) " :lib "-" :id 4 :doc "https://mlochbaum.github.io/BQN/help/scan.html" :sig "-"}
"BQN@˝ (insert)@-" {:lang "BQN" :algo "˝ (insert)" :lib "-" :id 1 :doc "https://mlochbaum.github.io/BQN/help/insert.html" :sig "-"}
"Kotlin@fold@collections" {:lang "Kotlin" :algo "fold" :lib "collections" :id 1 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/fold.html" :sig "-"}
"Kotlin@runningFold@collections" {:lang "Kotlin" :algo "runningFold" :lib "collections" :id 4 :doc "https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/running-fold.html" :sig "-"}
"D@fold@algorithm.iteration" {:lang "D" :algo "fold" :lib "algorithm.iteration" :id 1 :doc "https://dlang.org/library/std/algorithm/iteration/fold.html" :sig "-"}
"Scala@fold@various" {:lang "Scala" :algo "fold" :lib "various" :id 1 :doc "https://www.scala-lang.org/api/2.12.8/scala/collection/immutable/Vector.html#fold[A1%3E:A](z:A1)(op:(A1,A1)=%3EA1):A1" :sig "-"}
"Scala@reduce@various" {:lang "Scala" :algo "reduce" :lib "various" :id 1 :doc "https://www.scala-lang.org/api/2.12.8/scala/collection/immutable/Vector.html#reduce[A1%3E:A](op:(A1,A1)=%3EA1):A1" :sig "-"}
"Ruby@inject@Enumerable" {:lang "Ruby" :algo "inject" :lib "Enumerable" :id 1 :doc "https://apidock.com/ruby/Enumerable/inject" :sig "-"}
"Pharo@inject:into:@Collection" {:lang "Pharo" :algo "inject:into:" :lib "Collection" :id 1 :doc "" :sig "-"}
"Pharo@reduce:@Collection" {:lang "Pharo" :algo "reduce:" :lib "Collection" :id 1 :doc "" :sig "-"}
"Pharo@fold:@Collection" {:lang "Pharo" :algo "fold:" :lib "Collection" :id 1 :doc "" :sig "-"}
"BQN@´ (fold)@-" {:lang "BQN" :algo "´ (fold)" :lib "-" :id 1 :doc "https://mlochbaum.github.io/BQN/help/fold.html" :sig "-"}
"APL@⌿ (reduce first)@-" {:lang "APL" :algo "⌿ (reduce first)" :lib "-" :id 1 :doc "https://microapl.com/apl_help/ch_020_020_810.htm" :sig "-"}
"BQN@⌜ (table)@-" {:lang "BQN" :algo "⌜ (table)" :lib "-" :id 6 :doc "https://mlochbaum.github.io/BQN/help/table.html" :sig "-"}
"J@/ (table)@-" {:lang "J" :algo "/ (table)" :lib "-" :id 6 :doc "https://code.jsoftware.com/wiki/Vocabulary/slash#dyadic" :sig "-"}
"Rust@counts@Itertools" {:lang "Rust" :algo "counts" :lib "Itertools" :id 19 :doc "https://docs.rs/itertools/latest/itertools/trait.Itertools.html#method.counts" :sig "-"}
"C++@adjacent_transform@<ranges>" {:lang "C++" :algo "adjacent_transform" :lib "<ranges>" :id 5 :doc "https://en.cppreference.com/w/cpp/ranges/adjacent_transform_view" :sig "-"}
"Gleam@map@list" {:lang "Gleam" :algo "map" :lib "list" :id 2 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#map" :sig "-"}
"Gleam@range@list" {:lang "Gleam" :algo "range" :lib "list" :id 8 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#range" :sig "-"}
"Gleam@map2@list" {:lang "Gleam" :algo "map2" :lib "list" :id 30 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#map2" :sig "-"}
"Gleam@scan@list" {:lang "Gleam" :algo "scan" :lib "list" :id 4 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#scan" :sig "-"}
"Kap@⌻ (outer product)@-" {:lang "Kap" :algo "⌻ (outer product)" :lib "-" :id 6 :doc "https://kapdemo.dhsdevelopments.com/reference.html#_outer_product" :sig "-"}
"Uiua@⊞ (table)@-" {:lang "Uiua" :algo "⊞ (table)" :lib "-" :id 6 :doc "https://www.uiua.org/docs/tables" :sig "-"}
"Common Lisp@reduce@-" {:lang "Common Lisp" :algo "reduce" :lib "-" :id 1 :doc "https://novaspec.org/cl/f_reduce" :sig "-"}
"Common Lisp@map@-" {:lang "Common Lisp" :algo "map" :lib "-" :id 2 :doc "https://novaspec.org/cl/f_map" :sig "-"}
"Mathematica@Fold@-" {:lang "Mathematica" :algo "Fold" :lib "-" :id 1 :doc "https://reference.wolfram.com/language/ref/Fold.html" :sig "-"}
"Mathematica@/@ (Map)@-" {:lang "Mathematica" :algo "/@ (Map)" :lib "-" :id 2 :doc "http://reference.wolfram.com/language/ref/Map.html" :sig "-"}
"Mathematica@Select@-" {:lang "Mathematica" :algo "Select" :lib "-" :id 3 :doc "http://reference.wolfram.com/language/ref/Select.html" :sig "-"}
"Mathematica@FoldList@-" {:lang "Mathematica" :algo "FoldList" :lib "-" :id 4 :doc "http://reference.wolfram.com/language/ref/FoldList.html" :sig "-"}
"Mathematica@Outer@-" {:lang "Mathematica" :algo "Outer" :lib "-" :id 6 :doc "http://reference.wolfram.com/language/ref/Outer.html" :sig "-"}
"Mathematica@Inner@-" {:lang "Mathematica" :algo "Inner" :lib "-" :id 7 :doc "http://reference.wolfram.com/language/ref/Inner.html" :sig "-"}
"Mathematica@Range@-" {:lang "Mathematica" :algo "Range" :lib "-" :id 8 :doc "http://reference.wolfram.com/language/ref/Range.html" :sig "-"}
"Mathematica@Partition@-" {:lang "Mathematica" :algo "Partition" :lib "-" :id 11 :doc "http://reference.wolfram.com/language/ref/Partition.html" :sig "-"}
"Mathematica@DeleteAdjacentDuplicates@-" {:lang "Mathematica" :algo "DeleteAdjacentDuplicates" :lib "-" :id 14 :doc "http://reference.wolfram.com/language/ref/DeleteAdjacentDuplicates.html" :sig "-"}
"Mathematica@DeleteDuplicates@-" {:lang "Mathematica" :algo "DeleteDuplicates" :lib "-" :id 15 :doc "http://reference.wolfram.com/language/ref/DeleteDuplicates.html" :sig "-"}
"Mathematica@ToString@-" {:lang "Mathematica" :algo "ToString" :lib "-" :id 16 :doc "http://reference.wolfram.com/language/ref/ToString.html" :sig "-"}
"Mathematica@MemberQ@-" {:lang "Mathematica" :algo "MemberQ" :lib "-" :id 17 :doc "http://reference.wolfram.com/language/ref/MemberQ.html" :sig "-"}
"Mathematica@Counts@-" {:lang "Mathematica" :algo "Counts" :lib "-" :id 19 :doc "http://reference.wolfram.com/language/ref/Counts.html" :sig "-"}
"Mathematica@AnyTrue@-" {:lang "Mathematica" :algo "AnyTrue" :lib "-" :id 20 :doc "http://reference.wolfram.com/language/ref/AnyTrue.html" :sig "-"}
"Mathematica@Join@-" {:lang "Mathematica" :algo "Join" :lib "-" :id 21 :doc "http://reference.wolfram.com/language/ref/Join.html" :sig "-"}
"Mathematica@StringJoin@-" {:lang "Mathematica" :algo "StringJoin" :lib "-" :id 21 :doc "http://reference.wolfram.com/language/ref/StringJoin.html" :sig "-"}
"Mathematica@Catenate@-" {:lang "Mathematica" :algo "Catenate" :lib "-" :id 21 :doc "http://reference.wolfram.com/language/ref/Catenate.html" :sig "-"}
"Mathematica@Transpose@-" {:lang "Mathematica" :algo "Transpose" :lib "-" :id 22 :doc "http://reference.wolfram.com/language/ref/Transpose.html" :sig "-"}
"Mathematica@SplitBy@-" {:lang "Mathematica" :algo "SplitBy" :lib "-" :id 23 :doc "http://reference.wolfram.com/language/ref/SplitBy.html" :sig "-"}
"Mathematica@ConstantArray@-" {:lang "Mathematica" :algo "ConstantArray" :lib "-" :id 25 :doc "http://reference.wolfram.com/language/ref/ConstantArray.html" :sig "-"}
"Mathematica@MapThread@-" {:lang "Mathematica" :algo "MapThread" :lib "-" :id 30 :doc "http://reference.wolfram.com/language/ref/MapThread.html" :sig "-"}
"Mathematica@AllTrue@-" {:lang "Mathematica" :algo "AllTrue" :lib "-" :id 31 :doc "http://reference.wolfram.com/language/ref/AllTrue.html" :sig "-"}
"Mathematica@Append@-" {:lang "Mathematica" :algo "Append" :lib "-" :id 42 :doc "http://reference.wolfram.com/language/ref/Append.html" :sig "-"}
"Mathematica@AppendTo@-" {:lang "Mathematica" :algo "AppendTo" :lib "-" :id 42 :doc "http://reference.wolfram.com/language/ref/AppendTo.html" :sig "-"}
"Mathematica@RotateLeft@-" {:lang "Mathematica" :algo "RotateLeft" :lib "-" :id 4104 :doc "http://reference.wolfram.com/language/ref/RotateLeft.html" :sig "-"}
"Mathematica@RotateRight@-" {:lang "Mathematica" :algo "RotateRight" :lib "-" :id 4104 :doc "http://reference.wolfram.com/language/ref/RotateRight.html" :sig "-"}
"Common Lisp@every@-" {:lang "Common Lisp" :algo "every" :lib "-" :id 31 :doc "https://novaspec.org/cl/f_every" :sig "-"}
"Common Lisp@remove-duplicates@-" {:lang "Common Lisp" :algo "remove-duplicates" :lib "-" :id 15 :doc "https://novaspec.org/cl/f_remove-duplicates" :sig "-"}
"Common Lisp@apply@-" {:lang "Common Lisp" :algo "apply" :lib "-" :id 51 :doc "https://novaspec.org/cl/f_apply" :sig "-"}
"Haskell@intersperse@Data.List" {:lang "Haskell" :algo "intersperse" :lib "Data.List" :id 12 :doc "https://hackage.haskell.org/package/base/docs/Data-List.html#v:intersperse" :sig "-"}
"Rust@intersperse@itertools" {:lang "Rust" :algo "intersperse" :lib "itertools" :id 12 :doc "https://docs.rs/itertools/0.14.0/itertools/fn.intersperse.html" :sig "-"}
"Gleam@intersperse@list" {:lang "Gleam" :algo "intersperse" :lib "list" :id 12 :doc "https://hexdocs.pm/gleam_stdlib/gleam/list.html#intersperse" :sig "-"}
"Elixir@intersperse@Enum" {:lang "Elixir" :algo "intersperse" :lib "Enum" :id 12 :doc "https://hexdocs.pm/elixir/1.12/Enum.html#intersperse/2" :sig "-"}
"Racket@intersperse@seq" {:lang "Racket" :algo "intersperse" :lib "seq" :id 12 :doc "https://docs.racket-lang.org/seq/index.html#%28def._%28%28lib._seq%2Fmain..rkt%29._intersperse%29%29" :sig "-"}
"Racket@add-between@" {:lang "Racket" :algo "add-between" :lib "" :id 12 :doc "https://docs.racket-lang.org/reference/pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._add-between%29%29" :sig "-"}
"Elm@intersperse@core" {:lang "Elm" :algo "intersperse" :lib "core" :id 12 :doc "https://package.elm-lang.org/packages/elm/core/latest/List#intersperse" :sig "-"}
"Agda@foldl@various" {:lang "Agda" :algo "foldl" :lib "various" :id 1 :doc "https://agda.github.io/agda-stdlib/v2.2/Effect.Foldable.html#1843" :sig "-"}
"Agda@map@various" {:lang "Agda" :algo "map" :lib "various" :id 2 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#1634" :sig "-"}
"Agda@filter@various" {:lang "Agda" :algo "filter" :lib "various" :id 3 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#10687" :sig "-"}
"Agda@scanl@various" {:lang "Agda" :algo "scanl" :lib "various" :id 4 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Scans.Base.html#1105" :sig "-"}
"Agda@cartesianProductWith@Data.List" {:lang "Agda" :algo "cartesianProductWith" :lib "Data.List" :id 6 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#2132" :sig "-"}
"Agda@cartesianProduct@Data.List" {:lang "Agda" :algo "cartesianProduct" :lib "Data.List" :id 100 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#2317" :sig "-"}
"Agda@upTo@Data.List" {:lang "Agda" :algo "upTo" :lib "Data.List" :id 8 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#6689" :sig "-"}
"Agda@allFin@Data.Vec" {:lang "Agda" :algo "allFin" :lib "Data.Vec" :id 8 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Vec.Base.html#6708" :sig "-"}
"Agda@partitionᵇ@Data.List" {:lang "Agda" :algo "partitionᵇ" :lib "Data.List" :id 10 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#11170" :sig "-"}
"Agda@intersperse@Data.List" {:lang "Agda" :algo "intersperse" :lib "Data.List" :id 12 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#1819" :sig "-"}
"Agda@group@Data.Vec" {:lang "Agda" :algo "group" :lib "Data.Vec" :id 13 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Vec.Base.html#7275" :sig "-"}
"Agda@deduplicate@Data.List" {:lang "Agda" :algo "deduplicate" :lib "Data.List" :id 15 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#13320" :sig "-"}
"Agda@show@various" {:lang "Agda" :algo "show" :lib "various" :id 16 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Bool.Show.html#412" :sig "-"}
"Agda@_∈_@Data.List" {:lang "Agda" :algo "_∈_" :lib "Data.List" :id 17 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Membership.Setoid.html#925" :sig "-"}
"Agda@any@various" {:lang "Agda" :algo "any" :lib "various" :id 20 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#4857" :sig "-"}
"Agda@concat@various" {:lang "Agda" :algo "concat" :lib "various" :id 21 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#4350" :sig "-"}
"Agda@transpose@various" {:lang "Agda" :algo "transpose" :lib "various" :id 22 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.Vec.Base.html#9519" :sig "-"}
"Agda@repeat@Codata.Guarded.Stream" {:lang "Agda" :algo "repeat" :lib "Codata.Guarded.Stream" :id 24 :doc "https://agda.github.io/agda-stdlib/v2.2/Codata.Guarded.Stream.html#1216" :sig "-"}
"Agda@replicate@various" {:lang "Agda" :algo "replicate" :lib "various" :id 25 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#5319" :sig "-"}
"Agda@zipWith@various" {:lang "Agda" :algo "zipWith" :lib "various" :id 30 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#2734" :sig "-"}
"Agda@all@various" {:lang "Agda" :algo "all" :lib "various" :id 31 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#4910" :sig "-"}
"Agda@_++_@various" {:lang "Agda" :algo "_++_" :lib "various" :id 42 :doc "https://agda.github.io/agda-stdlib/v2.2/Data.List.Base.html#1734" :sig "-"}
"Kap@⍸ (where)@-" {:lang "Kap" :algo "⍸ (where)" :lib "-" :id 18 :doc "" :sig "-"}
"BQN@/ (indices)@-" {:lang "BQN" :algo "/ (indices)" :lib "-" :id 18 :doc "https://mlochbaum.github.io/BQN/doc/replicate.html#indices" :sig "-"}
"Uiua@⊚ (where)@-" {:lang "Uiua" :algo "⊚ (where)" :lib "-" :id 18 :doc "https://www.uiua.org/docs/where" :sig "-"}
"J@I. (indices)@-" {:lang "J" :algo "I. (indices)" :lib "-" :id 18 :doc "https://code.jsoftware.com/wiki/Vocabulary/icapdot" :sig "-"}
"Python@where@NumPy" {:lang "Python" :algo "where" :lib "NumPy" :id 18 :doc "https://numpy.org/doc/stable/reference/generated/numpy.where.html" :sig "-"}
"BQN@⍷ (deduplicate)@-" {:lang "BQN" :algo "⍷ (deduplicate)" :lib "-" :id 15 :doc "https://mlochbaum.github.io/BQN/doc/selfcmp.html#deduplicate" :sig "-"}
})
