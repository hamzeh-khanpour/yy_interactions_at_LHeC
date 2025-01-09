

import ROOT

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Use CMS style
hep.style.use("CMS")


def compare_Etall():
    # Open the ROOT file
    file = ROOT.TFile.Open("LHeC_tautau_E_10_10.root")
    if not file or file.IsZombie():
        print("Error: Cannot open the ROOT file.")
        return

    # Retrieve the TTrees
    tree1 = file.Get("LHeC_E")
    tree2 = file.Get("LHeC_E_750GeV")
    if not tree1 or not tree2:
        print("Error: One or both TTrees not found in the ROOT file.")
        return

    # Remove statistics globally
    ROOT.gStyle.SetOptStat(0)

    # Define integrated luminosity and cross-section values
    integrated_luminosity = 1.0  # fb^-1
    cross_section_LHeC_E = 4.55272144e+01  # pb
    cross_section_LHeC_E_750GeV = 3.28684083e+01  # pb

    # Calculate total entries
    nentries1 = tree1.GetEntries()
    nentries2 = tree2.GetEntries()

    # Calculate event weights
    event_weight1 = cross_section_LHeC_E * integrated_luminosity / nentries1
    event_weight2 = cross_section_LHeC_E_750GeV * integrated_luminosity / nentries2

    # Create histograms for Etall
    hist1 = ROOT.TH1F("hist1", "", 20, -10.0, 10.0)
    hist2 = ROOT.TH1F("hist2", "", 20, -10.0, 10.0)

    # Fill histograms with event weights
    tree1.Draw(f"Etall>>hist1", f"{event_weight1}")
    tree2.Draw(f"Etall>>hist2", f"{event_weight2}")


    # Set Y-axis range for both histograms
    hist1.GetYaxis().SetRangeUser(0.1, 10.0)
    hist2.GetYaxis().SetRangeUser(0.1, 10.0)



    # Set histogram styles
    hist1.SetLineColor(ROOT.kRed)
    hist1.SetLineWidth(3)
    # Set X and Y axis titles and their sizes
    hist1.GetXaxis().SetTitle("#eta^{#tau^{+}}")
    hist1.GetXaxis().SetTitleSize(0.04)  # Increase title size
    hist1.GetXaxis().SetLabelSize(0.03)  # Increase label size
    hist1.GetXaxis().SetLabelFont(62)  # Set bold font for numbers on X-axis
    hist1.GetXaxis().SetTitleFont(42)  # Keep the title in regular font

    hist1.GetYaxis().SetTitle("d#sigma/d#eta^{#tau^{+}} [pb/GeV]")
    hist1.GetYaxis().SetTitleSize(0.04)  # Increase title size
    hist1.GetYaxis().SetLabelSize(0.03)  # Increase label size
    hist1.GetYaxis().SetLabelFont(62)  # Set bold font for numbers on Y-axis
    hist1.GetYaxis().SetTitleFont(42)  # Keep the title in regular font

    hist1.GetYaxis().SetRangeUser(0.1, 10.0)  # Set Y-axis range

    hist2.SetLineColor(ROOT.kBlue)
    hist2.SetLineWidth(3)
    # Set X and Y axis titles and their sizes
    hist2.GetXaxis().SetTitle("#eta^{#tau^{+}}")
    hist2.GetXaxis().SetTitleSize(0.04)  # Increase title size
    hist2.GetXaxis().SetLabelSize(0.03)  # Increase label size
    hist2.GetXaxis().SetLabelFont(62)  # Set bold font for numbers on X-axis
    hist2.GetXaxis().SetTitleFont(42)  # Keep the title in regular font

    hist2.GetYaxis().SetTitle("d#sigma/d#eta^{#tau^{+}} [pb/GeV]")
    hist2.GetYaxis().SetTitleSize(0.04)  # Increase title size
    hist2.GetYaxis().SetLabelSize(0.03)  # Increase label size
    hist2.GetYaxis().SetLabelFont(62)  # Set bold font for numbers on Y-axis
    hist2.GetYaxis().SetTitleFont(42)  # Keep the title in regular font

    hist2.GetYaxis().SetRangeUser(0.1, 10.0)  # Set Y-axis range


    # Print integrals
    print(f"Integral of Etall (LHeC_E): {hist1.Integral()} pb")
    print(f"Integral of Etall (LHeC_E_750GeV): {hist2.Integral()} pb")

    # Create a canvas for the comparison
    c1 = ROOT.TCanvas("c1", "Etall Comparison", 800, 900)


# Adjust margins to leave space for axis titles and additional labels
    c1.SetLeftMargin(0.14)   # Similar to `left=0.15`
    c1.SetRightMargin(0.05)  # Similar to `right=0.95`
    c1.SetBottomMargin(0.10) # Similar to `bottom=0.12`
    c1.SetTopMargin(0.05)    # Similar to `top=0.95`

    c1.SetFrameLineWidth(3)

    # Draw histograms
    hist1.Draw("HIST")
    hist2.Draw("HIST SAME")

    # Add a legend with line styles and reduced spacing
    legend = ROOT.TLegend(0.50, 0.75, 0.9, 0.85)  # Adjust the vertical bounds
    legend.SetNColumns(1)  # Keep one column of entries
    legend.SetMargin(0.2)  # Adjust margin to make entries closer
    legend.AddEntry(hist1, "elastic", "l")  # Solid line for elastic
    legend.AddEntry(hist2, "elastic (0.750 TeV)", "l")  # Dashed line for inelastic

    legend.SetTextSize(0.040)  # Adjust text size for readability
    legend.SetTextFont(42)  # Set text font to Helvetica Medium
    legend.SetBorderSize(0)  # Remove border around the legend
    legend.SetFillStyle(0)   # Make the legend background transparent

    # Set line colors and styles for histograms
    hist1.SetLineColor(ROOT.kBlue)
    hist1.SetLineWidth(3)
    hist1.SetLineStyle(1)  # Solid line for elastic

    hist2.SetLineColor(ROOT.kRed)
    hist2.SetLineWidth(3)
    hist2.SetLineStyle(2)  # Dashed line for inelastic

    # Draw the legend
    legend.Draw()





    # Add additional information to the canvas
    label1 = ROOT.TLatex()
    label1.SetNDC()  # Normalize coordinates to canvas dimensions
    label1.SetTextSize(0.035)
    label1.SetTextFont(42)  # Set the font to Helvetica Medium (42)
    label1.DrawLatex(0.18, 0.87, "Q^{2}_{e} < 10 GeV^{2};  Q^{2}_{p} < 10 GeV^{2}")


    label2 = ROOT.TLatex()
    label2.SetNDC()
    label2.SetTextSize(0.035)
    label2.SetTextFont(42)  # Set the font to Helvetica Medium (42)
    label2.DrawLatex(0.18, 0.82, "M_{N} < 10 GeV")


    label3 = ROOT.TLatex()
    label3.SetNDC()
    label3.SetTextSize(0.035)
    label3.SetTextFont(42)  # Set the font to Helvetica Medium (42)
    label3.DrawLatex(0.18, 0.76, "W > 10 GeV")


    # Save the canvas
    c1.SaveAs("dSigmadEta_tau_W10GeV.pdf")
#    c1.SaveAs("dSigmadEta_tau_W10GeV.png")
#    c1.SaveAs("dSigmadEta_tau_W10GeV.root")

    # Clean up
    del hist1
    del hist2

    del label1
    del label2
    del label3

    del legend
    del c1
    file.Close()

if __name__ == "__main__":
    compare_Etall()
