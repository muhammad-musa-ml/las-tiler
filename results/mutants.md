# Mutants

Each row breaks one guard in a temporary copy of the code and runs the whole suite with -x.
Baseline before any change: 94 passed in 29.33s.

| what was broken | file | result | first failing test |
|---|---|---|---|
| tile cell subtraction done in int32 | las_tiler/tile.py | killed | test_coordinates_further_apart_than_an_int32_holds |
| a point on a tile edge goes to the lower tile | las_tiler/tile.py | killed | test_synth_tile_check_end_to_end |
| sort inside a chunk is not stable | las_tiler/tile.py | killed | test_every_point_lands_where_a_plain_python_loop_puts_it[0] |
| VLR block not copied into tiles | las_tiler/tile.py | killed | test_a_correct_cut_passes_and_reports_what_it_checked |
| each chunk overwrites the tile file | las_tiler/tile.py | killed | test_a_correct_cut_passes_and_reports_what_it_checked |
| tile size need not be a multiple of the scale | las_tiler/tile.py | killed | test_a_bad_tile_size_exits_2_with_a_message |
| tile size tolerance scales with the tile size, not the scale | las_tiler/tile.py | killed | test_tile_units_refuses_sizes_that_are_not_whole_multiples[1000.0000005-0.001] |
| a non-finite tile size is not turned into an error | las_tiler/tile.py | killed | test_an_infinite_tile_size_and_a_zero_chunk_exit_2 |
| a directory that already holds tiles is accepted | las_tiler/tile.py | killed | test_a_directory_that_already_holds_tiles_is_refused |
| a chunk size below one is accepted | las_tiler/tile.py | killed | test_a_chunk_size_below_one_is_refused[0] |
| a chunk size below one is accepted by the chunk reader | las_tiler/las.py | killed | test_a_chunk_size_below_one_is_refused_instead_of_looping_forever[0] |
| OSError is not turned into exit 2 | las_tiler/cli.py | killed | test_a_missing_source_exits_2_for_check_and_for_tile |
| an extent beyond 32-bit coordinates is accepted | las_tiler/synth.py | killed | test_an_extent_that_does_not_fit_32_bit_coordinates_is_refused |
| the major version is not compared | las_tiler/las.py | killed | test_refuses_another_major_version |
| per-return counts of a tile are not accumulated | las_tiler/tile.py | killed | test_one_changed_byte_is_found_when_every_count_still_agrees |
| file size is not compared with the header | las_tiler/las.py | killed | test_refuses_a_truncated_file |
| return number read from four bits | las_tiler/las.py | killed | test_return_counts_ignore_numbers_outside_one_to_five |
| record length is not compared with the format | las_tiler/las.py | killed | test_refuses_extra_bytes_per_record_even_when_the_file_size_agrees |
| digest is a plain sum with no mixing | las_tiler/check.py | killed | test_digest_ignores_order_and_sees_a_byte_moved_between_records |
| points outside their cell are not counted | las_tiler/check.py | killed | test_a_point_in_the_wrong_tile_is_found_when_only_one_index_is_wrong[iy] |
| only the column of a point is compared with its tile | las_tiler/check.py | killed | test_a_point_in_the_wrong_tile_is_found_when_only_one_index_is_wrong[ix] |
| only the row of a point is compared with its tile | las_tiler/check.py | killed | test_a_point_in_the_wrong_tile_is_found_when_only_one_index_is_wrong[iy] |
| digest cut to 32 bits | las_tiler/check.py | killed | test_a_correct_cut_passes_and_reports_what_it_checked |
| tile scale, offset and VLR count are not compared with the source | las_tiler/check.py | killed | test_a_tile_scale_or_offset_that_differs_from_the_source_is_found[131-scale] |
| manifest format, scale and offset are not compared with the source | las_tiler/check.py | killed | test_a_manifest_scale_that_differs_from_the_source_is_found |
| manifest source point count is not compared | las_tiler/check.py | killed | test_a_manifest_source_point_count_that_differs_is_found |
| a tile with no points is not flagged | las_tiler/check.py | killed | test_a_tile_file_with_no_points_is_found |
| a manifest that is not an object crashes | las_tiler/check.py | killed | test_a_manifest_of_the_wrong_shape_is_a_problem_and_not_a_crash[[]] |
| digest comparison skipped | las_tiler/check.py | killed | test_a_dropped_point_is_found |
| VLR comparison skipped | las_tiler/check.py | killed | test_a_changed_vlr_block_is_found |
| header bounds comparison skipped | las_tiler/check.py | killed | test_a_wrong_bounds_field_is_found[179] |
| a source with points and no tiles is not flagged | las_tiler/check.py | killed | test_a_manifest_with_no_tiles_does_not_pass_a_source_that_has_points |
| tile files missing from the manifest are not flagged | las_tiler/check.py | killed | test_a_stray_tile_file_is_found |

33 of 33 killed; 0 survived; 0 hung; 0 not run.
