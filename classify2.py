from keras.models import load_model
from array import array
from glob import glob
import pandas as pd
import collections
import uproot
import ROOT
import sys
from os import environ, path, mkdir, listdir
environ['KERAS_BACKEND'] = 'tensorflow'


def getGuess(df, index):
    """
    Try to grab the NN output value if there is one

    Parameters:
        df (pandas.DataFrame): all events in the current file
        index (int): event index to try and grab

    Returns:
        prob_sig (float): NN output value provided one exists for
        this index. Otherwise, returns -999
    """
    try:
        prob_sig = df.loc[index, 'prob_sig']
    except:
        prob_sig = -999
    return prob_sig

def getVariable(df, index, vrble):
    """to get variables form dataframe"""
    try:
        kine_var = df.loc[index, vrble]
    except:
        kine_var = -999999999999
    return kine_var

def build_filelist(input_dir):
    """
    Build list of files to process. Returns a map in case files need to be
    split into groups at a later stage of the analysis.

    Parameters:
        input_dir (string): name of input directory

    Returns:
        filelist (map): list of files that need to be processed
    """
    filelist = collections.defaultdict(list)
    filelist['all'] = [fname for fname in glob('{}/*.root'.format(input_dir))]
    return filelist


def main(args):
    model = load_model('models/{}.hdf5'.format(args.model))  # load the NN model with weights
    all_data = pd.HDFStore(args.input_name)  # load the rescaled data

    # make a place for output files if it doesn't exist
    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    filelist = build_filelist(args.input_dir)  # get files to process
    for _, files in filelist.items():
        for ifile in files:
            print ('Processing file: {}'.format(ifile))
            
            # now let's try and get this into the root file
            root_file = ROOT.TFile(ifile, 'READ')            

            # create output file and copy things input->output
            fname = ifile.split('/')[-1].replace('.root', '')
            fout = ROOT.TFile('{}/{}.root'.format(args.output_dir, fname), 'recreate')  # make new file for output
            fout.cd()
            #allEvents = root_file.Get('allEvents').Clone()
            #ana = root_file.Get('allEvents').Clone()
            #allEvents.Write()
            #ana.Write()

            data = all_data['nominal']  # load the correct tree (only nominal for now)

            # get dataframe for this sample
            sample = data[(data['sample_names'] == fname)]
            # drop all variables not going into the network
            to_classify = sample[[
		      	'Lepton_pt','Lepton_eta', 'LeadJet_pt','MET','ST','HT','nBTagMed_DeepFLV',
			'nCentralJets','DPHI_LepMet','DPHI_LepleadJet','DR_LepClosestJet',
			'bVsW_ratio','Angle_MuJet_Met'
            ]]

            # do the classification
            guesses = model.predict(to_classify.values, verbose=False)
            out = sample.copy()
            out['prob_sig'] = guesses[:, 0]
            out.set_index('idx', inplace=True)
            # copy the input tree to the new file
            itree = root_file.Get('Skim')
            ntree = itree.CloneTree(-1, 'fast')
                        
            t1 = ROOT.TTree('Mu_trees', 'sddsdsdfd')
            t2 = ROOT.TTree('Ele_trees','dfsfdfdsf')
            t3 = ROOT.TTree('NN_scores_Mu', 'dfhsfsdhfsf')
            t4 = ROOT.TTree('NN_scores_Ele', 'dfhsfsdf')
            # create a new branch and loop through guesses to fill the new tree.
            # this will be much easier once uproot supports writing TTrees
            NN_sig_mu = array('f', [0.])
            NN_sig_ele = array('f', [0.])
            disc_branch_sig_mu = t3.Branch('NN_disc', NN_sig_mu, 'NN_disc/F')
            disc_branch_sig_ele = t4.Branch('NN_disc', NN_sig_ele, 'NN_disc/F')
            evt_index = 0 
 
            weight_1 = array('f',[0.]) 
            weight_2 = array('f',[0.])
            weight_3 = array('f',[0.])
            weight_4 = array('f',[0.])

            # for Muon Trees
            lepPt_1 = array('f',[0.])
            lepEta_1 = array('f',[0.])
            leadjetPt_1 = array('f',[0.])
            met_1 = array('f',[0.])
            st_1 = array('f',[0.])
            st_v2_1 = array('f',[0.])
            dr_lepleadjet_1 = array('f',[0.])
            ht_1 = array('f',[0.])
            nBtagMeddeepFlv_1 = array('f',[0.])
            Ncentraljets_1 = array('f',[0.])
            dphilepMet_1 = array('f',[0.])
            dphilepleadjet_1 = array('f',[0.])
            drlepclosestjet_1 = array('f',[0.])
            bvswratio_1 = array('f',[0.])
            anglemujetmet_1 = array('f',[0.])
           
            t1.Branch('evtwt', weight_1, 'evtwt/F')
            t3.Branch('evtwt', weight_3, 'evtwt/F')
            t1.Branch('Lepton_pt', lepPt_1, 'Lepton_pt/F')
            t1.Branch('Lepton_eta', lepEta_1, 'Lepton_eta/F')
            t1.Branch('LeadJet_pt', leadjetPt_1, 'LeadJet_pt/F')
            t1.Branch('MET', met_1, 'MET/F')
            t3.Branch('ST', st_1, 'ST/F')#
            t3.Branch('ST_v2', st_v2_1, 'ST_v2/F')#
            t3.Branch('DR_LepLeadJet', dr_lepleadjet_1, 'DR_LepLeadJet/F')#
            t3.Branch('HT', ht_1, 'HT/F')#
            t3.Branch('nBTagMed_DeepFLV', nBtagMeddeepFlv_1, 'nBTagMed_DeepFLV/F')#
            t1.Branch('nCentralJets', Ncentraljets_1, 'nCentralJets/F')
            t1.Branch('DPHI_LepMet', dphilepMet_1, 'DPHI_LepMet/F')
            t1.Branch('DPHI_LepleadJet', dphilepleadjet_1, 'DPHI_LepleadJet/F')
            t1.Branch('DR_LepClosestJet', drlepclosestjet_1, 'DR_LepClosestJet/F')
            t1.Branch('bVsW_ratio', bvswratio_1, 'bVsW_ratio/F')
            t1.Branch('Angle_MuJet_Met', anglemujetmet_1, 'Angle_MuJet_Met/F')

            # for Electron Trees
            lepPt_2 = array('f',[0.])
            lepEta_2 = array('f',[0.])
            leadjetPt_2 = array('f',[0.])
            met_2 = array('f',[0.])
            st_2 = array('f',[0.])
            st_v2_2 = array('f',[0.])
            dr_lepleadjet_2 = array('f',[0.])
            ht_2 = array('f',[0.])
            nBtagMeddeepFlv_2 = array('f',[0.])
            Ncentraljets_2 = array('f',[0.])
            dphilepMet_2 = array('f',[0.])
            dphilepleadjet_2 = array('f',[0.])
            drlepclosestjet_2 = array('f',[0.])
            bvswratio_2 = array('f',[0.])
            anglemujetmet_2 = array('f',[0.])

            t2.Branch('evtwt', weight_2, 'evtwt/F')
            t4.Branch('evtwt', weight_4, 'evtwt/F')
            t2.Branch('Lepton_pt', lepPt_2, 'Lepton_pt/F')
            t2.Branch('Lepton_eta', lepEta_2, 'Lepton_eta/F')
            t2.Branch('LeadJet_pt', leadjetPt_2, 'LeadJet_pt/F')
            t2.Branch('MET', met_2, 'MET/F')
            t4.Branch('ST', st_2, 'ST/F')#
            t4.Branch('ST_v2', st_v2_2, 'ST_v2/F')#
            t4.Branch('DR_LepLeadJet', dr_lepleadjet_2, 'DR_LepLeadJet/F')#
            t4.Branch('HT', ht_2, 'HT/F')#
            t4.Branch('nBTagMed_DeepFLV', nBtagMeddeepFlv_2, 'nBTagMed_DeepFLV/F')#
            t2.Branch('nCentralJets', Ncentraljets_2, 'nCentralJets/F')
            t2.Branch('DPHI_LepMet', dphilepMet_2, 'DPHI_LepMet/F')
            t2.Branch('DPHI_LepleadJet', dphilepleadjet_2, 'DPHI_LepleadJet/F')
            t2.Branch('DR_LepClosestJet', drlepclosestjet_2, 'DR_LepClosestJet/F')
            t2.Branch('bVsW_ratio', bvswratio_2, 'bVsW_ratio/F')
            t2.Branch('Angle_MuJet_Met', anglemujetmet_2, 'Angle_MuJet_Met/F')
           
            '''if 'WJet' in fname:
	        out_copy = out[(out['st_v2'] > 500)&(out['ht'] > 500)&(out['nBtagMed_deepFLV'] == 0)&(out['dr_lepleadjet'] > 0.5)&(out['dr_lepclosestJet'] < 1.5)]
                print(out_copy.ht)
            elif 'TT' in fname:
                out_copy = out[(out['st_v2'] > 500)&(out['ht'] > 500)&(out['nBtagMed_deepFLV'] >= 2)&(out['dr_lepleadjet'] > 0.5)&(out['dr_lepclosestJet'] < 1.5)]           
            else:
                print('Hahaha! This dataset is either data or signal')
                out_copy = out'''
            out_copy = out[(out['st_v2'] > 500)&(out['ht'] > 500)&(out['nBtagMed_deepFLV'] == 0)&(out['dr_lepleadjet'] > 0.5)&(out['dr_lepclosestJet'] > -1)]#Wjet ctrl region
            #out_copy = out[(out['st_v2'] > 500)&(out['ht'] > 500)&(out['nBtagMed_deepFLV'] >= 2)&(out['dr_lepleadjet'] > 0.5)&(out['dr_lepclosestJet'] < 1.5)]#TTbar ctrl region
 

            for _ in range(len(out_copy)): #looping only over the indices in input df
                if evt_index % 100000 == 0: #prints for every 100000 events :)
                    print ('Processing: {}% completed'.format((evt_index*100)/ntree.GetEntries()))
               # print (out.index[evt_index], getVariable(out, out.index[evt_index], 'Event_Flag'), getVariable(out, out.index[evt_index], 'lep_pt'))  
                #evt_weight = getVariable(out, out.index[evt_index], 'EvtWt')

                if getVariable(out_copy, out_copy.index[evt_index], 'Event_Flag') == 13: #for Muon events
                    st_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'st')
                    st_v2_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'st_v2')
                    dr_lepleadjet_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'dr_lepleadjet')
                    ht_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'ht')
                    nBtagMeddeepFlv_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'nBtagMed_deepFLV')
                    weight_3[0] = getVariable(out_copy, out_copy.index[evt_index], 'EvtWt')
                    NN_sig_mu[0] = getGuess(out_copy, out_copy.index[evt_index])
                    if NN_sig_mu[0] < 0.9:
                        lepPt_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'lep_pt')
                        leadjetPt_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'leadjet_pt')
                        met_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'met')
                        lepEta_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'lep_eta')
                        Ncentraljets_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'Ncentraljets')
                        dphilepMet_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'dphi_lepMet')
                        dphilepleadjet_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'dphi_lepleadJet')
                        drlepclosestjet_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'dr_lepclosestJet')
                        bvswratio_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'bV_sW_ratio')
                        anglemujetmet_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'angle_mujet_met')
                        weight_1[0] = getVariable(out_copy, out_copy.index[evt_index], 'EvtWt')
                        #t1.SetWeight(evt_weight)
                        t1.Fill()
                    fout.cd()
                    #t3.SetWeight(evt_weight)
                    t3.Fill()
                    fout.cd()

                elif getVariable(out_copy, out_copy.index[evt_index], 'Event_Flag') == 11: #for Electron events
                    st_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'st')
                    st_v2_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'st_v2')
                    dr_lepleadjet_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'dr_lepleadjet')
                    ht_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'ht')
                    nBtagMeddeepFlv_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'nBtagMed_deepFLV')
                    weight_4[0] = getVariable(out_copy, out_copy.index[evt_index], 'EvtWt')
                    NN_sig_ele[0] = getGuess(out_copy, out_copy.index[evt_index])
                    if NN_sig_ele[0] < 0.9:
                        lepPt_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'lep_pt')
                        leadjetPt_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'leadjet_pt')
                        met_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'met')
                        lepEta_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'lep_eta')
                        Ncentraljets_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'Ncentraljets')
                        dphilepMet_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'dphi_lepMet')
                        dphilepleadjet_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'dphi_lepleadJet')
                        drlepclosestjet_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'dr_lepclosestJet')
                        bvswratio_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'bV_sW_ratio')
                        anglemujetmet_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'angle_mujet_met')
                        weight_2[0] = getVariable(out_copy, out_copy.index[evt_index], 'EvtWt')
                        #t2.SetWeight(evt_weight)
                        t2.Fill()
                    fout.cd()
                    #t4.SetWeight(evt_weight)
                    t4.Fill()
                    fout.cd()
                else:
                    print ('no Muon or Electron found!')
                evt_index += 1
                #fout.cd()
                #t3.Fill()
                #t4.Fill()
           
            fout.Write()
            root_file.Close()
            fout.Close()

           
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', action='store', dest='model', default='testModel', help='name of model to use')
    parser.add_argument('--input', '-i', action='store', dest='input_name', default='test', help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir', default='input_files/etau_stable_Oct24', help='name of ROOT input directory')
    parser.add_argument('--out', '-o', action='store', dest='output_dir', default='output_files/example')

    main(parser.parse_args())

