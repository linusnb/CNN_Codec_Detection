?	@?:s?*?@@?:s?*?@!@?:s?*?@	?a??????a?????!?a?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6@?:s?*?@?.??e@1???@?:w@AOu??p??I9{???@Y?MI??*	?O???R?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?hq?0?"@!?a???X@)v?ݑ?z"@1?A?e?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???I????!p?>?P??)???I????1p?>?P??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??u?+.??!oJ??ۍ??)??u?+.??1oJ??ۍ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism%?}?e???!?T/`dS??)E?
)????1??j??1??:Preprocessing2F
Iterator::ModelT5A?} ??!?O?????)?B:<??s?1???Ԓ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 31.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?a?????I?????@@Q??s??P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?.??e@?.??e@!?.??e@      ??!       "	???@?:w@???@?:w@!???@?:w@*      ??!       2	Ou??p??Ou??p??!Ou??p??:	9{???@9{???@!9{???@B      ??!       J	?MI???MI??!?MI??R      ??!       Z	?MI???MI??!?MI??b      ??!       JGPUY?a?????b q?????@@y??s??P@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??TD?:??!??TD?:??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?x?[????!??(??L??0"=
sequential/conv_layer2/Relu_FusedConv2D<LF곜??!z??t??"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$S?B???!?A?m???0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGradmb?t?2??!$??]????"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????g`??!????l???0"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGradD?{r1??!??aӃ??"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInputN?r????!G????_??0"=
sequential/conv_layer3/Relu_FusedConv2D{?2????!s??5???"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??_qա?!?$?ӌ???0Q      Y@YHeO?b2A@a\Mء?fP@q?L?.??/@y?E?]??Z?"?

both?Your program is POTENTIALLY input-bound because 31.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?15.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 