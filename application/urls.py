from django.conf.urls import url
from . import views

app_name = 'segment'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^register/$', views.register, name='register'),
    url(r'^login_user/$', views.login_user, name='login_user'),
    url(r'^logout_user/$', views.logout_user, name='logout_user'),
    url(r'^input_file/$', views.input_file, name='input_file'),
    url(r'^variable_select/$', views.var_select, name='var_select'),
    url(r'^process_file/(?P<option>[a-z]+)/$', views.process_file, name='process_file'),
    url(r'^table/$', views.show_table, name='show_table'),
    url(r'^chart/$', views.show_chart, name='show_chart'),
    url(r'^tree/$', views.decision_tree, name='show_tree'),
    url(r'^visualise/$', views.fscharts, name='visualise'),
    url(r'^boundaries/$', views.boxplot, name='boxplot'),
    url(r'^boundaries/(?P<sep>(.*)?)/$', views.boxplot_param, name='boxplot_param'),
    url(r'^distribution/(?P<cat>(.*)?)/$', views.multiplot_param, name='multiplot_param'),
    url(r'^distribution/$', views.multiplot, name='multiplot'),
    url(r'^features/$', views.features_selection, name='features'),
    url(r'^download/$', views.download, name='dwn'),
    url(r'^input_file/(?P<filename>(.*)?)/(?P<fileid>[0-9]+)/$', views.disp_data, name='data'),
]
